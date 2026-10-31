"""Whole-block structural configurations and compensation placement."""
import copy
import hashlib
import itertools
import json
import torch
from torch import nn
from .modeling import block, put, slots, logits
NORM = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def geometry(model, kind, x):
    shapes, handles = ({}, [])
    for i, (parent, index, _) in enumerate(slots(model, kind), 1):

        def hook(module, inputs, output, number=i):
            shapes[str(number)] = [list(inputs[0].shape[1:]), list(output.shape[1:])]
        handles.append(parent[index].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            logits(model, x[:2].to(next(model.parameters()).device), kind)
    finally:
        for handle in handles:
            handle.remove()

    def signature(module):
        return ([(n,
            type(m).__name__,
            m.extra_repr()) for n,
            m in module.named_modules()],
            [(n,
            list(t.shape)) for n,
            t in module.state_dict().items()])
    pairs = [[i,
        i + 1] for i in range(1,
        len(shapes)) if shapes[str(i)] == shapes[str(i + 1)] and signature(block(model,
        kind,
        i)) == signature(block(model,
        kind,
        i + 1))]
    return dict(shapes=shapes,
        removable_blocks=[int(i) for i,
        shape in shapes.items() if shape[0] == shape[1]],
        whole_adjacent_pairs=pairs)

def cases(kind, g, families=('drop',), max_blocks=4, scope='suffix'):
    modes = ['S', 'A', 'S+A'] if kind in ('bc8', 'tc14') else ['A']
    result = []

    def add(family, drops=(), copies=(), mode='F', placement='none'):
        c = dict(family=family,
            drops=list(drops),
            copies=[list(pair) for pair in copies],
            mode=mode,
            scope=placement)
        result.append(dict(case_id=digest(c)[:20], **c))
    add('baseline')
    for k in range(1, max_blocks + 1):
        for drops in itertools.combinations(g['removable_blocks'], k):
            if 'drop' in families:
                add('drop', drops=drops)
            if 'norm' in families:
                for mode in modes:
                    add('recovery', drops=drops, mode=mode, placement=scope)
            if 'affine' in families:
                add('gap_affine', drops=drops, mode='A', placement='gap')
        edges = sorted(g['whole_adjacent_pairs'] + [[b,
            a] for a,
            b in g['whole_adjacent_pairs']],
            key=lambda pair: (pair[1],
            pair[0]))
        for copies in itertools.combinations(edges, k):
            sources, targets = ({a for a, _ in copies}, [b for _, b in copies])
            if len(set(targets)) != k or sources & set(targets):
                continue
            if 'reuse' in families:
                add('reuse', copies=copies)
            if 'reuse_norm' in families:
                for mode in modes:
                    add('reuse_norm', copies=copies, mode=mode, placement='target_only')
    return result

class Affine(nn.Module):

    def __init__(self, width, axis):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.axis = axis

    def forward(self, x):
        shape = [1] * x.ndim
        shape[self.axis] = -1
        return x * self.weight.view(shape) + self.bias.view(shape)

def independent_norm(source):
    target = copy.deepcopy(source)
    norm_parameters = {id(p) for m in source.modules() if isinstance(m,
        NORM) for p in m.parameters(recurse=False)}
    for name, parameter in source.named_parameters():
        if id(parameter) not in norm_parameters:
            parent, _, leaf = name.rpartition('.')
            owner = target.get_submodule(parent) if parent else target
            owner._parameters[leaf] = parameter
    return target

def build_case(original, kind, case, g=None):
    model = copy.deepcopy(original).eval().requires_grad_(False)
    drops, copies = (case.get('drops', []), case.get('copies', []))
    selected = []
    sources, targets = ({a for a, _ in copies}, [b for _, b in copies])
    if g is not None:
        edges = {tuple(pair) for pair in g['whole_adjacent_pairs']}
        edges |= {(b, a) for a, b in edges}
    for source, target in copies:
        replacement = block(model, kind, source)
        if case['family'] == 'reuse_norm':
            replacement = independent_norm(replacement)
            selected.extend((m for m in replacement.modules() if isinstance(m, NORM)))
        put(model, kind, target, replacement)
    for number in drops:
        replacement = nn.Identity()
        if case['family'] == 'gap_affine':
            axis = 1 if kind in ('bc8', 'tc14') else -1
            width = g['shapes'][str(number)][0][0 if axis == 1 else -1]
            replacement = Affine(width, axis).to(next(model.parameters()).device)
            selected.append(replacement)
        put(model, kind, number, replacement)
    if case['family'] == 'recovery':
        if case.get('scope', 'suffix') == 'all':
            selected = [m for m in model.modules() if isinstance(m, NORM)]
        elif drops:
            parts = [block(model,
                kind,
                i) for i in range(min(drops) + 1,
                len(slots(model,
                kind)) + 1) if i not in drops]
            if kind == 'bc8':
                parts.append(model.classifier)
            if kind == 'kwmlp':
                parts.append(model.to_logits)
            selected = [m for part in parts for m in part.modules() if isinstance(m, NORM)]
    ids = {id(m) for m in selected}
    chosen = {name: m for name, m in model.named_modules() if id(m) in ids}
    return (model.eval().requires_grad_(False), chosen)
