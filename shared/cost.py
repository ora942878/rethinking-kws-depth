"""Observed-graph operation counts for recovery and inference experiments.

Adapted from experiment/tables/recovery_cost/op_counter.py in the research
workspace. The counter is self-contained and reads no experiment results.

Core counts are dense contraction MACs (one MAC = two FLOPs), NOT half
the total estimate. Non-core floating-point work uses explicit scalar
arithmetic formulas; it is reported separately because fused kernel
implementations may execute different instructions. Unknown operators
are retained in the report, never silently called zero-cost.

Example:
    with OperationCounter() as counter:
        with counter.phase("forward"):
            loss = criterion(model(x), y)
        with counter.phase("backward"):
            loss.backward()
        with counter.phase("optimizer"):
            optimizer.step()
    report = counter.report()


No wall-clock measurements are made.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from math import prod
from typing import Any
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry

def _tensors(value):
    return [x for x in tree_flatten(value)[0] if isinstance(x, torch.Tensor)]

def _numel(value):
    return sum((int(x.numel()) for x in _tensors(value)))

def _floating(value):
    return any((x.is_floating_point() or x.is_complex() for x in _tensors(value)))

def _argument(func, args, kwargs, name, default=None):
    """Resolve positional/keyword arguments using the operator schema."""
    if name in kwargs:
        return kwargs[name]
    for i, field in enumerate(func._schema.arguments):
        if field.name == name:
            return args[i] if i < len(args) else default
    return default

def _conv_macs(x, weight, output, transposed=False):
    spatial = x.shape[2:] if transposed else output.shape[2:]
    return int(x.shape[0] * prod(spatial) * weight.numel())

def _adaptive_pool_connections(input_shape, output_shape):
    """Number of input/output bin connections, including overlapping bins."""
    extent_sums = []
    for ins, outs in zip(input_shape[2:], output_shape[2:]):
        extent_sums.append(sum((((i + 1) * ins + outs - 1) // outs - i * ins // outs for i in range(outs))))
    return int(prod(input_shape[:2]) * prod(extent_sums))

def _core_count(func, args, kwargs, out):
    """Return MACs, extra non-core FLOPs, formula, or None."""
    name = func._schema.name.split('::')[-1]
    get = lambda key, default=None: _argument(func, args, kwargs, key, default)
    if name in {'convolution', '_convolution', 'conv1d', 'conv2d', 'conv3d'}:
        x, w = args[:2]
        transposed = bool(get('transposed', False))
        bias = get('bias', None)
        return (_conv_macs(x,
            w,
            out,
            transposed),
            out.numel() if bias is not None else 0,
            'dense grouped convolution; padded positions counted; bias add separate')
    if name == 'convolution_backward':
        grad, x, w = args[:3]
        mask = list(get('output_mask'))
        transposed = bool(get('transposed', False))
        base = _conv_macs(x, w, grad, transposed)
        macs = base * (int(mask[0]) + int(mask[1]))
        bias = grad.numel() - grad.shape[1] if mask[2] else 0
        return (macs, int(bias), 'convolution backward: forward MACs times requested dx/dW')
    if name in {'mm', 'addmm', 'bmm', 'baddbmm'}:
        a, b = args[1:3] if name in {'addmm', 'baddbmm'} else args[:2]
        macs = int(out.numel() * a.shape[-1])
        extra = 0
        if name in {'addmm', 'baddbmm'}:
            beta, alpha = (get('beta', 1), get('alpha', 1))
            if beta != 0:
                extra += int(out.numel()) * (1 + int(beta != 1))
            if alpha != 1:
                extra += int(out.numel())
        return (macs, extra, 'matrix contraction; alpha/beta and additive input separate')
    if name in {'mv', 'dot', 'vdot', 'matmul', 'linear'}:
        a, b = args[:2]
        k = a.shape[-1]
        extra = out.numel() if name == 'linear' and get('bias') is not None else 0
        return (int(out.numel() * k), int(extra), 'dense vector/matrix contraction')
    packet = func._overloadpacket
    if packet in flop_registry and ('attention' in name or name == '_scaled_mm'):
        flop = int(flop_registry[packet](*args, **kwargs, out_val=out))
        return (flop // 2, 0, 'PyTorch contraction formula; fused non-matmul arithmetic excluded')
    return None
_ZERO = {'alias',
    'detach',
    'detach_',
    'view',
    '_unsafe_view',
    'reshape',
    'flatten',
    'transpose',
    'transpose_',
    't',
    'permute',
    'unsqueeze',
    'unsqueeze_',
    'squeeze',
    'squeeze_',
    'expand',
    'expand_as',
    'as_strided',
    'as_strided_',
    'slice',
    'select',
    'split',
    'split_with_sizes',
    'unbind',
    'chunk',
    'narrow',
    'contiguous',
    'clone',
    'copy_',
    '_to_copy',
    'to',
    'lift_fresh',
    'lift_fresh_copy',
    'empty',
    'empty_like',
    'empty_strided',
    'new_empty',
    'new_empty_strided',
    'zeros',
    'zeros_like',
    'new_zeros',
    'ones',
    'ones_like',
    'new_ones',
    'full',
    'full_like',
    'new_full',
    'zero_',
    'fill_',
    'resize_',
    'resize_as_',
    'set_',
    'cat',
    'stack',
    'index',
    'index_select',
    'gather',
    'scatter_',
    'slice_scatter',
    'select_scatter',
    'slice_backward',
    'select_backward',
    'constant_pad_nd',
    'reflection_pad1d',
    'reflection_pad2d',
    'replication_pad1d',
    'arange',
    'scalar_tensor',
    'item',
    '_local_scalar_dense',
    'eq',
    'ne',
    'gt',
    'ge',
    'lt',
    'le',
    'isfinite',
    'isinf',
    'isnan',
    'all',
    'any',
    'argmax',
    'argmin',
    'max',
    'min',
    'amax',
    'amin',
    'clamp',
    'clamp_',
    'clamp_min',
    'clamp_min_',
    'clamp_max',
    'clamp_max_',
    'maximum',
    'minimum',
    'where',
    'relu',
    'relu_',
    'threshold',
    'threshold_backward',
    'masked_fill',
    'masked_fill_',
    'sign',
    'sgn',
    'resolve_conj',
    'conj',
    '_conj',
    'resolve_neg',
    'real',
    'imag',
    '_foreach_zero_',
    '_foreach_copy_',
    'record_stream'}

def _noncore_count(func, args, kwargs, out):
    """Return explicit arithmetic estimate + convention, or None."""
    name = func._schema.name.split('::')[-1]
    get = lambda key, default=None: _argument(func, args, kwargs, key, default)
    n = _numel(out)
    if name in _ZERO:
        return (0, 'memory/shape/index/comparison operation; no floating-point arithmetic counted')
    if not _floating((args, kwargs, out)):
        return (0, 'integer/boolean-only operation excluded')
    binary = {'add',
        'add_',
        'sub',
        'sub_',
        'mul',
        'mul_',
        'div',
        'div_',
        'true_divide',
        'true_divide_',
        'floor_divide',
        'remainder'}
    unary = {'neg',
        'neg_',
        'abs',
        'abs_',
        'exp',
        'exp_',
        'expm1',
        'log',
        'log_',
        'log1p',
        'sqrt',
        'sqrt_',
        'rsqrt',
        'rsqrt_',
        'reciprocal',
        'sin',
        'cos',
        'tanh',
        'tanh_',
        'erf',
        'erfc',
        'pow',
        'pow_',
        'square'}
    if name in binary:
        factor = 1 + int(name.startswith(('add', 'sub')) and get('alpha', 1) != 1)
        return (n * factor, 'one scalar arithmetic operation per element; alpha scaling counted')
    if name in unary:
        return (n, 'one operation per scalar, including transcendental functions/powers')
    if name in {'addcmul', 'addcmul_', 'addcdiv', 'addcdiv_', 'lerp', 'lerp_'}:
        return (3 * n, 'three scalar operations per element (multiply/divide, scale, add)')
    if name.startswith('_foreach_'):
        stem = name.removeprefix('_foreach_').rstrip('_')
        if stem in {'add', 'sub', 'mul', 'div', 'sqrt', 'neg', 'reciprocal', 'exp', 'pow'}:
            count = _numel(args[0])
            factor = 1 + int(stem in {'add', 'sub'} and get('alpha', 1) != 1)
            return (factor * count, 'foreach elementwise arithmetic over tensor list')
        if stem in {'addcmul', 'addcdiv', 'lerp'}:
            return (3 * _numel(args[0]), 'foreach three-operation elementwise arithmetic')
    if name in {'sum', 'mean', 'prod'}:
        ins, outs = (_numel(args[0]), n)
        return (max(0,
            ins - outs) + (outs if name == 'mean' else 0),
            'reduction additions/multiplications plus mean division')
    if name in {'index_put', 'index_put_', '_index_put_impl_'}:
        if not get('accumulate', False):
            return (0, 'indexed assignment without accumulation; memory-only')
        indices = get('indices', args[1])
        present = [(i, x) for i, x in enumerate(indices) if x is not None]
        if any((x.dtype in {torch.bool, torch.uint8} for _, x in present)):
            return None
        broadcast = torch.broadcast_shapes(*(x.shape for _, x in present))
        selected = {i for i, _ in present}
        remaining = prod((s for i, s in enumerate(args[0].shape) if i not in selected))
        return (int(prod(broadcast) * remaining),
            'indexed accumulation: one addition per indexed destination contribution')
    if name in {'index_add', 'index_add_'}:
        return (int(get('source', args[3]).numel()), 'index_add: one addition per source contribution')
    if name in {'var', 'var_mean', 'std', 'std_mean'}:
        ins = _numel(args[0])
        groups = _numel(out[0] if isinstance(out, tuple) else out)
        return (4 * ins + (groups if name.startswith('std') else 0),
            'two-pass mean/variance estimate; std adds square root')
    if name in {'linalg_vector_norm', 'norm'}:
        order = get('ord', get('p', 2))
        ins = _numel(args[0])
        if order == 2:
            return (2 * ins, 'L2 squares + reduction + square root')
        if order == 1:
            return (2 * ins - n, 'L1 absolute values + reduction')
        return None
    if name == '_foreach_norm':
        p = args[1] if len(args) > 1 else 2
        if p == 2:
            return (2 * _numel(args[0]), 'foreach L2 norm')
        return None
    if name in {'_softmax', 'softmax', '_log_softmax', 'log_softmax'}:
        inp = args[0]
        dim = int(get('dim', -1))
        groups = inp.numel() // inp.shape[dim]
        return (4 * inp.numel() - (0 if 'log' in name else groups),
            'stable softmax/log-softmax; max comparisons excluded; exp/log counted as one')
    if name in {'_softmax_backward_data', '_log_softmax_backward_data'}:
        inp = args[0]
        dim = int(get('dim', -1))
        groups = inp.numel() // inp.shape[dim]
        return (4 * inp.numel() - groups, 'softmax/log-softmax backward scalar formula')
    if name in {'nll_loss_forward', 'nll_loss2d_forward', 'nll_loss_backward', 'nll_loss2d_backward'}:
        if get('weight') is not None:
            return None
        target = get('target')
        batch = int(target.numel())
        reduction = int(get('reduction', 1))
        if 'backward' in name:
            return (batch * (2 if reduction == 1 else 1),
                'unweighted NLL backward; assumes targets are not ignored')
        return (batch + (batch - 1 if reduction else 0) + int(reduction == 1),
            'unweighted NLL forward; assumes targets are not ignored')
    if name in {'batch_norm',
        'native_batch_norm',
        '_native_batch_norm_legit',
        '_native_batch_norm_legit_no_training',
        'cudnn_batch_norm',
        'miopen_batch_norm',
        '_batch_norm_with_update',
        '_batch_norm_no_update'}:
        x = get('input', args[0])
        N, C = (x.numel(), x.shape[1])
        train = bool(get('training', name == '_batch_norm_with_update'))
        if name in {'_native_batch_norm_legit_no_training', '_batch_norm_no_update'}:
            train = False
        affine = int(get('weight') is not None) + int(get('bias') is not None)
        if train:
            stats = 7 * C if get('running_mean') is not None else 0
            return ((5 + affine) * N + 2 * C + stats,
                'BN training two-pass arithmetic + affine + running-stat EMA estimate')
        return ((2 + affine) * N + 2 * C, 'BN inference scalar formula + invstd setup')
    if name in {'dropout', 'dropout_', 'feature_dropout', 'feature_dropout_'}:
        train, probability = (get('train'), get('p'))
        if train is False or probability == 0:
            return (0, 'dropout disabled (eval or p=0): identity, no arithmetic')
        return None
    if name == 'pad':
        if get('mode', 'constant') in {'constant', 'reflect', 'replicate', 'circular'}:
            return (0, 'padding fills/copies/indexes values; no floating-point arithmetic')
        return None
    if name in {'native_batch_norm_backward',
        'cudnn_batch_norm_backward',
        'miopen_batch_norm_backward',
        'batch_norm_backward'}:
        native = name == 'native_batch_norm_backward'
        x = get('input', args[1] if native else args[0])
        N, C = (x.numel(), x.shape[1])
        train = bool(get('train', True))
        mask = get('output_mask', [True, True, True])
        if native:
            dx, dw, db = map(bool, mask)
        else:
            dx, dw, db = [t is not None for t in out[:3]]
        if train and dx:
            return (9 * N + 3 * C,
                'BN training backward estimate; shared reductions reused for affine gradients')
        value = N + C if dx else 0
        value += (4 * N - C if dw else 0) + (N - C if db else 0)
        return (value, 'BN requested backward outputs; affine reductions and optional dx')
    if name in {'native_layer_norm', 'layer_norm'}:
        x = args[0]
        width = int(prod(get('normalized_shape', args[1])))
        N, groups = (x.numel(), x.numel() // width)
        affine = int(get('weight') is not None) + int(get('bias') is not None)
        return ((5 + affine) * N + 2 * groups,
            'LN two-pass mean/variance + normalization + affine scalar estimate')
    if name == 'native_layer_norm_backward':
        x = get('input', args[1])
        width = int(prod(get('normalized_shape', args[2])))
        N, groups = (x.numel(), x.numel() // width)
        dx, dw, db = map(bool, get('output_mask', args[-1]))
        value = 9 * N + 3 * groups if dx else 0
        value += (4 * N - width if dw else 0) + (N - width if db else 0)
        return (value, 'LN requested backward outputs; independent dx/dgamma/dbeta formula estimate')
    if name in {'gelu', 'gelu_backward'}:
        approx = get('approximate', 'none')
        factor = (9 if approx == 'tanh' else 5) if name == 'gelu' else 13 if approx == 'tanh' else 8
        return (factor * n, 'GELU scalar expression estimate; transcendental function counts as one')
    if name in {'sigmoid', 'sigmoid_backward', 'tanh_backward'}:
        return ((4 if name == 'sigmoid' else 3) * n, 'activation scalar expression estimate')
    if name in {'silu', 'silu_', 'silu_backward'}:
        return ((9 if name == 'silu_backward' else 5) * n,
            'SiLU scalar formula; sigmoid recomputed in backward; exp counted as one')
    if name in {'avg_pool1d',
        'avg_pool2d',
        'avg_pool3d',
        'avg_pool1d_backward',
        'avg_pool2d_backward',
        'avg_pool3d_backward'}:
        kernel = get('kernel_size')
        if isinstance(kernel, int):
            kernel = [kernel]
        if 'backward' in name:
            return (int(args[0].numel() * prod(kernel)), 'average-pool backward nominal kernel arithmetic')
        return (int(n * prod(kernel)), 'average-pool nominal kernel sum/division arithmetic')
    if name in {'_adaptive_avg_pool2d',
        '_adaptive_avg_pool3d',
        'adaptive_avg_pool1d',
        'adaptive_avg_pool2d',
        'adaptive_avg_pool3d'}:
        x = args[0]
        edges = _adaptive_pool_connections(x.shape, out.shape)
        return (edges, 'adaptive average pooling: bin additions plus division; overlapping bins included')
    if name in {'_adaptive_avg_pool2d_backward', '_adaptive_avg_pool3d_backward'}:
        edges = _adaptive_pool_connections(out.shape, args[0].shape)
        return (edges + max(0,
            edges - n),
            'adaptive average-pool backward: per-connection scaling plus overlapping contributions')
    return None

class OperationCounter(TorchDispatchMode):
    """Collect actual dispatched operators, labelled by an arbitrary phase."""

    def __init__(self):
        super().__init__()
        self.current_phase = 'unlabelled'
        self.reset()

    def reset(self):
        self._operators = defaultdict(dict)
        self._unsupported = defaultdict(dict)

    @contextmanager
    def phase(self, name):
        previous = self.current_phase
        self.current_phase = str(name)
        try:
            yield self
        finally:
            self.current_phase = previous

    def _unknown(self, name, out, reason):
        row = self._unsupported[self.current_phase].setdefault(name,
            dict(calls=0,
            output_elements=0,
            reason=reason))
        row['calls'] += 1
        row['output_elements'] += _numel(out)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        opname = str(func)
        try:
            core = _core_count(func, args, kwargs, out)
            if core is not None:
                macs, extra, formula = core
                if 'fused non-matmul' in formula:
                    self._unknown(opname + '::non_matmul_part', out, formula)
            else:
                noncore = _noncore_count(func, args, kwargs, out)
                if noncore is None:
                    self._unknown(opname, out, 'no arithmetic formula registered')
                    return out
                extra, formula = noncore
                macs = 0
        except Exception as exc:
            self._unknown(opname, out, f'formula error: {type(exc).__name__}: {exc}')
            return out
        row = self._operators[self.current_phase].setdefault(opname,
            dict(calls=0,
            core_flops=0,
            macs=0,
            estimated_noncore_flops=0,
            formula=formula,
            observed_shapes=[]))
        row['calls'] += 1
        row['macs'] += int(macs)
        row['core_flops'] += int(2 * macs)
        row['estimated_noncore_flops'] += int(extra)
        shapes = dict(inputs=[list(t.shape) for t in _tensors((args,
            kwargs))],
            outputs=[list(t.shape) for t in _tensors(out)])
        if shapes not in row['observed_shapes']:
            row['observed_shapes'].append(shapes)
        return out

    def report(self):
        phases = {}
        totals = dict(core_flops=0, macs=0, estimated_noncore_flops=0, estimated_total_flops=0)
        for phase in sorted(set(self._operators) | set(self._unsupported)):
            ops = self._operators[phase]
            row = {key: sum((value.get(key,
                0) for value in ops.values())) for key in ('core_flops',
                'macs',
                'estimated_noncore_flops')}
            row['estimated_total_flops'] = row['core_flops'] + row['estimated_noncore_flops']
            for key in totals:
                totals[key] += row[key]
            row['operators'] = dict(sorted(ops.items()))
            row['unsupported'] = dict(sorted(self._unsupported[phase].items()))
            row['uncovered_operator_calls'] = sum((v['calls'] for v in row['unsupported'].values()))
            phases[phase] = row
        totals['uncovered_operator_calls'] = sum((v['uncovered_operator_calls'] for v in phases.values()))
        return dict(phases=phases,
            totals=totals,
            conventions={'macs': 'Only convolution, matrix/vector products, and their dispatched backward contractions.',
            'core_flops': 'Two FLOPs per dense contraction MAC; padding counted at nominal dense shape.',
            'estimated_noncore_flops': 'Explicit scalar formulas; not exact hardware instruction counts.',
            'estimated_total_flops': 'Core + estimated noncore; excludes all unsupported arithmetic.',
            'excluded': ['memory/index/shape operations',
            'comparisons',
            'integer operations',
            'Python scalar arithmetic (including Adam bias-correction scalars)',
            'kernel launch/setup',
            'data preprocessing outside measured context'],
            'transcendentals': 'Each exp/log/sqrt/erf/tanh/power counts as one scalar operation.',
            'fused_attention': 'Core PyTorch formulas counted; fused softmax/dropout/scaling omitted and flagged.',
            'normalization': 'Two-pass reference arithmetic estimates; CUDA fused implementations can differ.',
            'time': 'Not measured.'})

def inference_cost(model, inputs, forward=None):
    """Count one eval-only, batch-one forward from already prepared features.

    The callback may unwrap model output or select the canonical logits. No
    frontend, loss, backward, optimizer, recovery or latency is included.
    """
    model.eval().requires_grad_(False)
    with torch.inference_mode(), OperationCounter() as counter:
        with counter.phase('forward'):
            output = (forward or model)(inputs)
    report = counter.report()
    return dict(parameters=sum((p.numel() for p in model.parameters())),
        batch_size=1,
        input_shape=list(inputs.shape),
        output_shape=list(output.shape),
        cost={key: report['totals'][key] for key in ('core_flops',
        'macs',
        'estimated_noncore_flops',
        'estimated_total_flops')},
        operator_report=report)
