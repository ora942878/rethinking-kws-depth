"""Measure paired-checkpoint recovery budgets, update time and operator costs.

The default recipes are one five-shot-per-class batch, 100 batches of 128,
and one full-training epoch for FT. Preparation, warm-up, cost tracing and
evaluation are outside the synchronized update timer. Each trial starts from
an independent copy of the explicitly supplied checkpoint.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
METHODS = ('F', 'S', 'A', 'S+A', 'inserted_affine', 'FT')
BUDGETS = ('fewshot_1', 'full_pool_100', 'full_epoch')
METRICS = ('core_flops', 'macs', 'estimated_noncore_flops', 'estimated_total_flops')
DEFAULT_DROPS = {'bc8': [4, 7, 8, 12], 'tc14': [2, 4], 'kwt1': [2, 5, 8, 12], 'kwmlp': [2, 5, 8, 12]}

def _backend():
    global torch, nn, F, load_model, load_prepared, build_case, geometry
    global configure_recovery, evaluate, logits, OperationCounter
    import torch
    from torch import nn
    from torch.nn import functional as F
    from shared.modeling import load_model, logits
    from shared.data import load_prepared
    from shared.structure import build_case, geometry
    from shared.recovery import configure_recovery
    from shared.metrics import evaluate
    from shared.cost import OperationCounter

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)

def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)

def schedule(budget, seed, labels, classes):
    """Return CPU indices; paired methods and arms use the same seed and rows."""
    labels = labels.detach().cpu()
    generator = torch.Generator().manual_seed(seed)
    if budget == 'fewshot_1':
        parts = []
        for label in range(classes):
            candidates = torch.where(labels == label)[0]
            parts.append(candidates[torch.randperm(len(candidates), generator=generator)[:5]])
        indices = torch.cat(parts)
        return [indices]
    order = torch.randperm(len(labels), generator=generator)
    if budget == 'full_pool_100':
        order = order[:12800]
    return list(order.split(128))

def budget_methods(budget, methods, explicit_methods=False):
    if budget == 'full_epoch':
        return ['FT']
    return [m for m in methods if m != 'F' and (m != 'FT' or budget == 'full_pool_100')]

def configure(original, kind, drops, method, momentum, g):
    case = dict(family='gap_affine' if method == 'inserted_affine' else 'recovery',
        drops=list(drops),
        copies=[],
        mode='A' if method == 'inserted_affine' else method,
        scope='gap' if method == 'inserted_affine' else 'suffix')
    if method in ('F', 'FT'):
        case.update(family='drop', scope='none')
    model, norms = build_case(original, kind, case, g)
    if method == 'FT':
        norms = {name: module for name,
            module in model.named_modules() if isinstance(module,
            (nn.modules.batchnorm._BatchNorm,
            nn.LayerNorm))}
    mode = 'A' if method == 'inserted_affine' else method
    params, optimizer = configure_recovery(model, norms, mode, momentum=momentum, lr=0.001)
    return (model, norms, params, optimizer)

def update(model, kind, x, y, params, optimizer, counter=None):
    phase = counter.phase if counter is not None else lambda _: nullcontext()
    if optimizer:
        optimizer.zero_grad(set_to_none=True)
    with phase('forward'), torch.set_grad_enabled(bool(params)):
        prediction = logits(model, x, kind)
    loss = None
    if optimizer:
        with phase('loss'):
            loss = F.cross_entropy(prediction, y)
        with phase('backward'):
            loss.backward()
        with phase('optimizer'):
            nn.utils.clip_grad_norm_(params, 1.0, foreach=False)
            optimizer.step()
    return loss

def _active(method, norms, params):
    return method != 'F' and bool(norms or params)

def trace_cost(original, kind, drops, method, momentum, sizes, train, g):
    model, norms, params, optimizer = configure(original, kind, drops, method, momentum, g)
    totals, traces, phase_totals = ({key: 0 for key in METRICS}, [], {})
    active = _active(method, norms, params)
    plan = [('first',
        sizes[0],
        1)] + [('steady',
        size,
        count) for size,
        count in sorted(Counter(sizes[1:]).items())] if active else []
    for role, size, multiplicity in plan:
        with OperationCounter() as counter:
            update(model, kind, train[0][:size], train[1][:size], params, optimizer, counter)
        report = counter.report()
        for key in METRICS:
            totals[key] += report['totals'][key] * multiplicity
        for phase, values in report['phases'].items():
            projected = phase_totals.setdefault(phase, {key: 0 for key in METRICS})
            for key in METRICS:
                projected[key] += values[key] * multiplicity
        traces.append(dict(role=role, batch_size=size, multiplicity=multiplicity, report=report))
    return dict(totals=totals,
        projected_by_phase=phase_totals,
        traces=traces,
        steps=len(sizes) if active else 0,
        sample_exposures=sum(sizes) if active else 0,
        trainable_parameters=sum((p.numel() for p in params)),
        model_parameters=sum((p.numel() for p in model.parameters())),
        cost_scope='Updates only; first step plus steady representatives weighted by actual batch sizes',
        conventions='Core FLOPs = 2 x contraction MACs; noncore FLOPs are explicit scalar estimates, not hardware instructions')

def warmup(original, kind, drops, method, momentum, sizes, train, g, steps, device):
    sync(device)
    started = time.perf_counter()
    model, norms, params, optimizer = configure(original, kind, drops, method, momentum, g)
    if _active(method, norms, params):
        for size in sorted(set(sizes)):
            for _ in range(steps):
                update(model, kind, train[0][:size], train[1][:size], params, optimizer)
    sync(device)
    return time.perf_counter() - started

def trial(original, kind, drops, method, momentum, indices, train, test, g, device, measure_time=True):
    sync(device)
    setup_start = time.perf_counter()
    model, norms, params, optimizer = configure(original, kind, drops, method, momentum, g)
    active_indices = [index.to(device) for index in indices] if _active(method, norms, params) else []
    sync(device)
    setup_seconds = time.perf_counter() - setup_start if measure_time else None
    cuda = torch.device(device).type == 'cuda'
    events = (torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True)) if cuda and measure_time else None
    if events:
        events[0].record()
        events[1].record()
        sync(device)
    started = time.perf_counter()
    if events:
        events[0].record()
    loss = None
    for index in active_indices:
        loss = update(model,
            kind,
            train[0].index_select(0,
            index),
            train[1].index_select(0,
            index),
            params,
            optimizer)
    if events:
        events[1].record()
    sync(device)
    update_seconds = time.perf_counter() - started if active_indices else 0.0
    gpu_seconds = events[0].elapsed_time(events[1]) / 1000 if events and active_indices else 0.0 if cuda else None
    if not measure_time:
        update_seconds = gpu_seconds = None
    model.eval().requires_grad_(False)
    started = time.perf_counter()
    metrics, _ = evaluate(model, kind, *test, batch_size=128)
    sync(device)
    evaluation_seconds = time.perf_counter() - started if measure_time else None
    return dict(test=metrics,
        steps=len(active_indices),
        sample_exposures=sum((len(i) for i in active_indices)),
        trainable_parameters=sum((p.numel() for p in params)),
        model_parameters=sum((p.numel() for p in model.parameters())),
        recovery_seconds=update_seconds,
        gpu_stream_seconds=gpu_seconds,
        timing_measured=measure_time,
        setup_seconds=setup_seconds,
        evaluation_seconds=evaluation_seconds,
        selected_norms=list(norms))

def summarize(rows):
    summary = []
    for budget, method in sorted({(row['budget'], row['method']) for row in rows}):
        group = [r for r in rows if (r['budget'], r['method']) == (budget, method)]
        first = group[0]
        item = {key: first[key] for key in ('model',
            'arm',
            'budget',
            'method',
            'deleted_blocks',
            'steps',
            'sample_exposures',
            'trainable_parameters',
            'model_parameters')}
        item['repetitions'] = len(group)
        accuracy = [r['test']['accuracy_percent'] for r in group]
        item.update(accuracy_mean_percent=statistics.mean(accuracy),
            accuracy_std_pp=statistics.stdev(accuracy) if len(accuracy) > 1 else 0.0)
        for name in ('recovery_seconds', 'gpu_stream_seconds', 'setup_seconds', 'evaluation_seconds'):
            values = [r[name] for r in group if r[name] is not None]
            item[name + '_mean'] = statistics.mean(values) if values else None
            item[name + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        for metric in METRICS:
            item[metric] = first['cost'][metric]
        item['estimated_gflops'] = item['estimated_total_flops'] / 1000000000.0
        item['gmacs'] = item['macs'] / 1000000000.0
        summary.append(item)
    return summary

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=tuple(DEFAULT_DROPS), required=True)
    p.add_argument('--arm', choices=['lambda_0', 'lambda_0p1'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prepared-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    return p

def main():
    args = parser().parse_args()
    _backend()
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    original = load_model(args.model, args.checkpoint, device)
    data = load_prepared(args.prepared_dir, device)
    g = geometry(original, args.model, data['test'][0])
    drops = DEFAULT_DROPS[args.model]
    methods = ['S', 'A', 'S+A', 'FT'] if args.model in ('bc8', 'tc14') else ['A', 'FT']
    classes = 35 if args.model == 'kwmlp' else 12
    records = []
    stream = (out / 'results.jsonl').open('w')

    def save(row):
        records.append(row)
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(row['budget'],
            row['method'],
            row['test']['accuracy_percent'],
            row['recovery_seconds'],
            flush=True)
    for name, deletion in [('original', []), ('direct', drops)]:
        result = trial(original, args.model, deletion, 'F', 0.1, [], data['train'], data['test'], g, device)
        save(dict(model=args.model,
            arm=args.arm,
            budget=name,
            method='F',
            deleted_blocks=deletion,
            repeat=0,
            seed=None,
            cost={k: 0 for k in METRICS},
            **result))
    for budget in BUDGETS:
        schedules = [schedule(budget, 20261007 + i, data['train'][1].cpu(), classes) for i in range(3)]
        sizes = [len(x) for x in schedules[0]]
        momentum = 1.0 if budget == 'fewshot_1' else 0.1
        for method in budget_methods(budget, methods):
            counted = trace_cost(original, args.model, drops, method, momentum, sizes, data['train'], g)
            warmup(original, args.model, drops, method, momentum, sizes, data['train'], g, 3, device)
            for i, indices in enumerate(schedules):
                result = trial(original,
                    args.model,
                    drops,
                    method,
                    momentum,
                    indices,
                    data['train'],
                    data['test'],
                    g,
                    device)
                save(dict(model=args.model,
                    arm=args.arm,
                    budget=budget,
                    method=method,
                    deleted_blocks=drops,
                    repeat=i + 1,
                    seed=20261007 + i,
                    cost=counted['totals'],
                    **result))
    stream.close()
    atomic(out / 'summary.json', summarize(records))
if __name__ == '__main__':
    main()
