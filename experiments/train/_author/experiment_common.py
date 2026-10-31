"""Shared monitoring and the unchanged channel-moment objective."""
from pathlib import Path
import json
import os
import time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAIRS = {'bc8': [(2,
    1),
    (4,
    3),
    (6,
    5),
    (7,
    5),
    (8,
    5),
    (10,
    9),
    (11,
    9),
    (12,
    9)],
    'tc14': [(2,
    1),
    (4,
    3),
    (6,
    5)],
    'kwt1': [(i,
    1) for i in range(2,
    13)],
    'kwmlp': [(i,
    1) for i in range(2,
    13)]}

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def dataset_root():
    from shared.paths import data_root
    return data_root()

def manifest(kind):
    return json.loads((ROOT / 'configs/manifests' / ('gsc35.json' if kind == 'kwmlp' else 'gsc12.json')).read_text(encoding='utf-8'))

def source_info(kind):
    entry = json.loads((ROOT / 'configs/sources.json').read_text(encoding='utf-8'))[kind]
    return dict(repository=entry['repository'], commit=entry['commit'])

class Monitor:
    """Status methods for the TC14 paired monitor, whose wrapper sets run fields."""

    def update(self, status='RUNNING', **kw):
        self.values.update(kw)
        elapsed = time.time() - self.started
        atomic(self.out / 'status.json',
            dict(**self.values,
            status=status,
            model=self.kind,
            pid=os.getpid(),
            started=self.started,
            updated=time.time(),
            step=self.step,
            total=self.total,
            epoch=self.step / self.spe,
            elapsed_seconds=elapsed,
            eta_seconds=max(0.0,
            elapsed / self.step * (self.total - self.step)) if self.step else None))

    def record(self, event, **kw):
        with (self.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event, step=self.step, time=time.time(), **kw)) + '\n')
        self.update(**kw)

    def finish(self, **kw):
        self.update('COMPLETE', **kw)
        atomic(self.out / 'COMPLETE.json', json.loads((self.out / 'status.json').read_text()))

class TorchMoments:

    def __init__(self, model, kind):
        import torch
        self.torch = torch
        self.model = model
        self.kind = kind
        self.values = {}
        self.handles = []
        blocks = [b for stage in model.BCBlocks for b in stage] if kind == 'bc8' else list(model.layers)
        self.handles.append(model.register_forward_pre_hook(lambda m, x: self.values.clear()))
        for n, b in enumerate(blocks, 1):

            def hook(m, inp, out, n=n):
                if not model.training or not torch.is_grad_enabled():
                    return
                axis = 1 if kind == 'bc8' else out.ndim - 1
                dims = tuple((i for i in range(out.ndim) if i != axis))
                self.values[n] = torch.var_mean(out.float(), dim=dims, correction=0)
            self.handles.append(b.register_forward_hook(hook))

    def loss(self):
        eps = 1e-05
        losses = []
        for n, ref in PAIRS[self.kind]:
            if n not in self.values or ref not in self.values:
                continue
            v, m = self.values[n]
            vr, mr = self.values[ref]
            losses.append(((m - mr).square() + ((v + eps).sqrt() - (vr + eps).sqrt()).square()).div(v + vr + eps).mean())
        if not losses:
            return (next(self.model.parameters()).sum() * 0, 0)
        return (self.torch.stack(losses).mean(), len(losses))
