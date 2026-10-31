"""Budgeted recovery with explicit update scope and frozen feature-extraction weights."""
import time
import torch
from torch import nn
from torch.nn import functional as F
from .modeling import logits

def configure_recovery(model, norms, mode, momentum=0.1, lr=0.001):
    model.eval().requires_grad_(False)
    stats = mode in ('S', 'S+A', 'FT')
    if mode == 'FT':
        model.requires_grad_(True)
        selected = dict(model.named_modules())
    else:
        selected = norms
        if mode in ('A', 'S+A'):
            for m in selected.values():
                for parameter in m.parameters(recurse=False):
                    parameter.requires_grad_(True)
    for m in selected.values():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = momentum
            m.train(stats)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, foreach=False, fused=False) if params else None
    return (params, optimizer)

def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)

def recover(model,
    kind,
    norms,
    case,
    train,
    steps,
    device='cpu',
    batch_size=128,
    lr=0.001,
    momentum=0.1,
    clip=1.0):
    mode = case['mode']
    params, optimizer = configure_recovery(model, norms, mode, momentum, lr)
    actual = steps if mode != 'F' and (norms or mode == 'FT') else 0
    synchronize(device)
    start = time.perf_counter()
    for i in range(actual):
        x = train[0][i * batch_size:(i + 1) * batch_size].to(device)
        y = train[1][i * batch_size:(i + 1) * batch_size].to(device)
        with torch.set_grad_enabled(bool(params)):
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            z = logits(model, x, kind)
            if optimizer:
                loss = F.cross_entropy(z, y)
                loss.backward()
                nn.utils.clip_grad_norm_(params, clip, foreach=False, error_if_nonfinite=True)
                optimizer.step()
    synchronize(device)
    elapsed = time.perf_counter() - start
    model.eval().requires_grad_(False)
    return dict(recovery_steps=actual,
        recovery_seconds=elapsed,
        batch_size=batch_size,
        processed_training_entries=actual * batch_size,
        learned_parameters=sum((p.numel() for p in params)),
        selected_norms=list(norms),
        no_applicable_norm=not bool(norms or mode == 'FT'),
        timing_scope='Update loop including batch transfer; excludes model construction and evaluation')
