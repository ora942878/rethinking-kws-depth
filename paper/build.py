'Generate the numerical paper figures and tables from saved results.'
from pathlib import Path
import csv, hashlib, importlib.util, json, statistics as st
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'outputs/paper'
NAMES = dict(bc8='BC-ResNet-8', tc14='TC-ResNet-14', kwt1='KWT-1', kwmlp='KW-MLP')
ARMS = dict(lambda_0='Reference', lambda_0p1='Regularized')
COUNTS = dict(bc8=[8, 28, 56, 70], tc14=[3, 3, 1], kwt1=[12, 66, 220, 495], kwmlp=[12, 66, 220, 495])
DATA = {}
BASE = {}
BUDGET = {}
FACTS = {}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))



def method(r):
    return 'Interface A' if r['family'] == 'gap_affine' else r['mode']

def rows(m, a='lambda_0', kind='F', k=None):
    return [r for r in DATA[m,
        a] if (r['family'] == 'drop' if kind == 'F' else r['family'] == 'gap_affine' if kind == 'Interface A' else r['family'] == 'reuse' if kind == 'reuse' else r['family'] == 'recovery' and r['scope'] == 'suffix' and (r['mode'] == kind)) and (k is None or r['k'] == k)]

def stats(rr):
    values = [r['accuracy_percent'] for r in rr]
    return dict(n=len(rr),
        mean=st.mean(values),
        median=st.median(values),
        min=min(values),
        max=max(values),
        mean_f1=st.mean((r['macro_f1'] for r in rr)))

def load():
    DATA.clear()
    BASE.clear()
    BUDGET.clear()
    FACTS.clear()
    for entry in read(ROOT / 'configs/checkpoints.json'):
        m, a = (entry['model'], entry['arm'])
        rr = [json.loads(s) for s in (ROOT / entry['results']).read_text(encoding='utf-8').splitlines() if s.strip()]
        DATA[m, a] = rr
        BASE[m, a] = next((r for r in rr if r['family'] == 'baseline'))
        BUDGET[m, a] = read(ROOT / entry['budget'])
        FACTS[m] = FACTS.get(m, {})
        modes = ['F', 'A', 'Interface A'] + (['S', 'S+A'] if m in ('bc8', 'tc14') else [])
        FACTS[m][a] = dict(baseline=BASE[m,
            a]['accuracy_percent'],
            stats={mode: [stats(rows(m,
            a,
            mode,
            k)) for k in range(1,
            len(COUNTS[m]) + 1)] for mode in modes})
    for sub in ('fig', 'table', 'doc'):
        (PAPER / sub).mkdir(parents=True, exist_ok=True)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def figures(selected=None):
    p = module(ROOT / 'paper/structure_style.py', 'structure_style')
    import matplotlib.pyplot as plt
    p.style()
    sources = {m: dict(baseline_accuracy=BASE[m, 'lambda_0']['accuracy_percent']) for m in NAMES}
    for number, name in [(1, 'direct_block_sensitivity'), *p.NAMES.items()]:
        if selected is not None and number not in selected:
            continue
        groups = {m: [BASE[m,
            'lambda_0']] + (rows(m,
            kind='F',
            k=1) if number == 1 else rows(m,
            kind='reuse',
            k=1) if number == 2 else rows(m,
            kind='F') if number == 3 else rows(m,
            kind='reuse') if number == 4 else rows(m,
            kind='F') + rows(m,
            kind='A') + (rows(m,
            kind='S') + rows(m,
            kind='S+A') if m in ('bc8',
            'tc14') else [])) for m in NAMES}
        if number == 1:
            fig, axes = p.horizontal_figure([1.05, 0.78, 1.15, 1.15])
            low = min((r['accuracy_change_pp'] for rr in groups.values() for r in rr))
            bottom = min(-10, np.floor(low / 10) * 10)
            for i, (ax, (m, title)) in enumerate(zip(axes, NAMES.items())):
                p.header(ax, i, title, 'Reference', sources[m]['baseline_accuracy'])
                rr = rows(m, kind='F', k=1)
                x = [r['drops'][0] for r in rr]
                ax.scatter(x, [r['accuracy_change_pp'] for r in rr], s=17, c='#0072B2', edgecolors='none')
                ax.set_xticks(x)
                ax.set_xlim(min(x) - 0.6, max(x) + 0.6)
                ax.set_ylim(bottom, 1)
                ax.set_yticks(np.arange(bottom, 1, 10))
                ax.set_xlabel('Block position')
                ax.axhline(0, c='#999999', lw=0.6)
            axes[0].set_ylabel('Accuracy change, $\\Delta$ (pp)')
        else:
            fig = p.plot_single(groups, sources) if number == 2 else p.plot_multi(number, groups, sources)
        flat = [r for rr in groups.values() for r in rr if r['family'] != 'baseline']
        low, high = fig.axes[0].get_ylim()
        field = 'accuracy_percent' if number == 5 else 'accuracy_change_pp'
        for ext in ('pdf', 'svg', 'png'):
            fig.savefig(PAPER / f'fig/{name}.{ext}', dpi=600, facecolor='white')
        plt.close(fig)
        fields = ['model',
            'arm',
            'case_id',
            'family',
            'k',
            'drops',
            'copies',
            'mode',
            'scope',
            'accuracy_percent',
            'accuracy_change_pp']
        with (PAPER / f'doc/{name}.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(({k: r[k] for k in fields} for r in flat))
    if selected is not None:
        return
    paired_figures()

def paired_figures(number=None):
    q = module(ROOT / 'paper/paired_style.py', 'paired_style')
    q.PAPER = PAPER
    groups = []
    points = []
    for m, title in NAMES.items():
        g = dict(model=m,
            title=title,
            counts=COUNTS[m],
            reference_full=BASE[m,
            'lambda_0']['accuracy_percent'],
            regularized_full=BASE[m,
            'lambda_0p1']['accuracy_percent'],
            statistics=[])
        for a, variant in ARMS.items():
            for mode, ss in FACTS[m][a]['stats'].items():
                for k, s in enumerate(ss, 1):
                    g['statistics'].append(dict(variant=variant,
                        k=k,
                        method=mode,
                        n=s['n'],
                        mean_accuracy=s['mean']))
                    points.extend((dict(model=m,
                        variant=variant,
                        k=k,
                        method=mode,
                        case_id=r['case_id'],
                        drops=r['drops'],
                        accuracy=r['accuracy_percent'],
                        change_pp=r['accuracy_change_pp']) for r in rows(m,
                        a,
                        mode,
                        k)))
        groups.append(g)
    if number in (None,7): q.plot(groups, points, 'deletion')
    if number in (None,8): q.plot(groups, points, 'recovery')
    q.table(groups)
    (PAPER / 'doc/regularized_deletion_data.json').write_text(json.dumps(dict(groups=groups,
        points=points),
        indent=2),
        encoding='utf-8')

def table(name, caption, label, columns, header, body, note=''):
    text = '\n'.join(['% Final matched results: generated by paper_refresh/refresh.py.',
        '\\begin{table}[htbp]',
        '\\caption{' + caption + '}',
        '\\label{' + label + '}',
        '\\centering\\small',
        '\\setlength{\\tabcolsep}{3pt}',
        '\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}' + columns + '@{}}',
        '\\toprule',
        header,
        '\\midrule',
        *body,
        '\\bottomrule',
        '\\end{tabular*}',
        '\\par\\smallskip\\begin{minipage}{\\linewidth}\\footnotesize',
        note,
        '\\end{minipage}',
        '\\end{table}',
        ''])
    (PAPER / f'table/{name}.tex').write_text(text, encoding='utf-8')

def line(values):
    return ' & '.join(map(str, values)) + ' \\\\'

def blocks(v):
    return '$\\{' + ','.join(map(str, v)) + '\\}$' if v else '$\\varnothing$'

def source_csv(name, records):
    with (PAPER / f'doc/{name}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def tables():
    count_body = []
    for m, title in NAMES.items():
        counts = [len(rows(m, kind=mode, k=k)) for mode in ('F', 'reuse') for k in range(1, 5)]
        count_body.append(line([title, *[f'{n:,}' for n in counts]]))
    table('structure_summary',
        'Number of configurations for block deletion and whole-block reuse.',
        'tab:structure_summary',
        'lrrrrrrrr',
        '\\multirow{2}{*}{Model} & \\multicolumn{4}{c}{Deletion} & \\multicolumn{4}{c}{Whole-block reuse} \\\\' + '\n' + '\\cmidrule(lr){2-5}\\cmidrule(l){6-9}' + '\n' + ' & $k=1$ & $k=2$ & $k=3$ & $k=4$ & $k=1$ & $k=2$ & $k=3$ & $k=4$ \\\\',
        count_body,
        '$k$ counts deleted blocks or reuse targets; zero denotes no compatible configuration. Counts apply to each checkpoint.')
    body = []
    FACTS['spaced'] = {}
    for m, title in NAMES.items():
        b = BASE[m, 'lambda_0']
        selected = [b]
        for k in range(2, len(COUNTS[m]) + 1):
            pool = [r for r in rows(m,
                kind='F',
                k=k) if all((y - x > 1 for x,
                y in zip(r['drops'],
                r['drops'][1:])))]
            selected.append(max(pool, key=lambda r: (r['accuracy_percent'], tuple((-i for i in r['drops'])))))
        FACTS['spaced'][m] = [{k: r[k] for k in ['drops',
            'accuracy_percent',
            'macro_f1',
            'parameters']} for r in selected]
        for i, r in enumerate(selected):
            acc = f"{r['accuracy_percent']:.2f}"
            acc = '\\textbf{' + acc + '}' if r['accuracy_percent'] > 90 else acc
            f1 = f"{r['macro_f1']:.4f}"
            if i:
                s = stats(rows(m, kind='F', k=len(r['drops'])))
                acc += f"~({s['mean']:.2f})"
                f1 += f"~({s['mean_f1']:.4f})"
            label = '\\multirow{' + str(len(selected)) + '}{*}{\\shortstack[l]{' + title + '\\\\(Reference)}}' if i == 0 else ''
            body.append(line([label,
                blocks(r['drops']),
                f"{r['parameters']:,}",
                f"{100 * (1 - r['parameters'] / b['parameters']):.2f}",
                acc,
                f1]))
        body.append('\\midrule')
    table('spaced_deletion',
        'Spaced deletion examples selected by test accuracy.',
        'tab:spaced_deletion',
        'llrrrr',
        'Model & Deleted blocks & Parameters & Saved (\\%) & Acc. (\\%) & Macro-F1 \\\\',
        body[:-1],
        'Examples maximize test accuracy among nonconsecutive sets of each size. Parentheses give means over all compatible sets, including consecutive sets. Bold marks accuracy above 90\\%; $\\varnothing$ denotes the complete model.')
    source_csv('spaced_deletion_cases',
        [dict(model=m,
        arm='lambda_0',
        **r) for m,
        rr in FACTS['spaced'].items() for r in rr])
    body = []
    for m, title in NAMES.items():
        b = BASE[m, 'lambda_0']['accuracy_percent']
        count = len(COUNTS[m])
        for k in range(1, count + 1):
            values = {mode: stats(rows(m,
                kind=mode,
                k=k))['mean'] for mode in ['F',
                'S',
                'A',
                'S+A'] if rows(m,
                kind=mode,
                k=k)}
            cells = []
            for mode in ['F', 'S', 'A', 'S+A']:
                if mode not in values:
                    cells.append('\\textemdash')
                    continue
                v = values[mode]
                acc = f'{v:.2f}'
                if v == max(values.values()):
                    acc = '\\textbf{' + acc + '}'
                cells.append(acc + f'~(${v - b:+.2f}$)')
            label = '\\multirow{' + str(count) + '}{*}{\\shortstack[l]{' + title + f'\\\\(Reference)\\\\{b:.2f}\\%}}}}' if k == 1 else ''
            body.append(line([label, k, *cells]))
        body.append('\\midrule')
    table('recovery_mean_gap',
        'Mean accuracy (\\%) and change from the complete model (pp) after deletion and recovery.',
        'tab:recovery_mean_gap',
        'lcrrrr',
        'Model & $k$ & F & S & A & S+A \\\\',
        body[:-1],
        'Means cover all compatible deletion sets; parentheses show changes from the complete model (pp). Bold marks the highest mean per row; dashes denote inapplicable settings.')
    source_csv('recovery_mean_gap',
        [dict(model=m,
        arm='lambda_0',
        method=mode,
        k=k,
        baseline=BASE[m,
        'lambda_0']['accuracy_percent'],
        **s) for m in NAMES for mode,
        ss in FACTS[m]['lambda_0']['stats'].items() if mode != 'Interface A' for k,
        s in enumerate(ss,
        1)])
    body = []
    FACTS['affine_parameters'] = {}
    for m, title in NAMES.items():
        widths = {r['drops'][0]: r['recovery']['learned_parameters'] for r in rows(m,
            kind='Interface A',
            k=1)}
        FACTS['affine_parameters'][m] = []
        for k in range(1, len(COUNTS[m]) + 1):
            ins = rows(m, kind='Interface A', k=k)
            norm = rows(m, kind='A', k=k)
            ip = st.mean((r['recovery']['learned_parameters'] for r in ins))
            merged = st.mean((sum((widths[d] for d in r['drops'] if d - 1 not in r['drops'])) for r in ins))
            ap = st.mean((r['recovery']['learned_parameters'] for r in norm))
            FACTS['affine_parameters'][m].append(dict(k=k, insert=ip, merged=merged, norm=ap))
            body.append(line([title if k == 1 else '',
                k,
                *[f"{stats(rows(m,
                kind=mode,
                k=k))['mean']:.2f}" for mode in ['F',
                'Interface A',
                'A']],
                f'{ip:,.0f} / {merged:,.0f}',
                f'{ap:,.0f}']))
        body.append('\\midrule')
    table('gap_affine_compact',
        'Mean accuracy and parameter counts for inserted affine layers and normalization affine adaptation.',
        'tab:gap_affine',
        'llrrrrr',
        'Model & Deleted & F (\\%) & Ins.\\ A (\\%) & A (\\%) & \\shortstack{$\\overline{P}_{\\mathrm{insert}}$\\\\train / merged} & $\\overline{P}_{\\mathrm{norm}}$ \\\\',
        body[:-1],
        'Means cover all compatible deletion sets; parameter counts are rounded to integers. Merged counts combine consecutive inserted affine layers (Equation~\\ref{eq:affine_parameter_counts}).')
    body = []
    for m, title in NAMES.items():
        bb = BUDGET[m, 'lambda_0']
        direct = next((r for r in bb if r['budget'] == 'direct'))
        orig = BASE[m, 'lambda_0']['accuracy_percent']
        body.append('\\multicolumn{8}{l}{\\textit{' + title + ' (Reference)}, delete ' + blocks(direct['deleted_blocks']) + f"; complete {orig:.2f}\\%, F {direct['accuracy_mean_percent']:.2f}\\%}} \\\\")
        for budget in ['fewshot_1', 'full_pool_100', 'full_epoch']:
            for mode in ['S', 'A', 'S+A', 'FT']:
                matches = [r for r in bb if r['budget'] == budget and r['method'] == mode]
                if not matches:
                    continue
                r, = matches
                label = mode if mode != 'FT' else 'Full (100)' if budget == 'full_pool_100' else 'Full (epoch)'
                body.append(line([label,
                    f"{r['sample_exposures']:,}",
                    r['steps'],
                    f"{r['trainable_parameters']:,}",
                    f"${r['accuracy_mean_percent']:.2f}\\pm{r['accuracy_std_pp']:.2f}$",
                    f"{r['estimated_gflops']:.2f}",
                    f"{r['gmacs']:.2f}",
                    f"${r['recovery_seconds_mean']:.4f}\\pm{r['recovery_seconds_std']:.4f}$"]))
        body.append('\\midrule')
    table('matched_recovery_budget',
        'Accuracy, computation, and measured recovery time under three update budgets.',
        'tab:matched_recovery_budget',
        'lrrrrrrr',
        'Method & Entries & Steps & Trainable & Acc. (\\%) & GFLOPs & GMACs & Time (s) \\\\',
        body[:-1],
        'Accuracy and time are means $\\pm$ sample standard deviations over three trials. Entries count sample exposures; FLOPs/MACs cover the full update schedule, and time covers the synchronized recovery loop.')
    body = []
    FACTS['inference'] = {}
    for m, title in NAMES.items():
        path = ROOT / f'results/inference/{m}.json'
        rr = read(path)
        FACTS['inference'][m] = rr
        for i, r in enumerate(rr):
            c = r['cost']
            body.append(line([title if i == 0 else '',
                'Complete' if i == 0 else 'Delete ' + blocks(r['deleted_blocks']),
                f"{r['parameters']:,}",
                *[f'{c[key] / 1000000000.0:.4f}' for key in ['macs',
                'core_flops',
                'estimated_noncore_flops',
                'estimated_total_flops']]]))
        body.append('\\midrule')
    table('inference_cost',
        'Single-sample network inference costs from prepared features before and after deletion.',
        'tab:inference_cost',
        'llrrrrr',
        'Model & Structure & Parameters & GMACs & \\shortstack{Core\\\\GFLOPs} & \\shortstack{Est. non-core\\\\GFLOPs} & \\shortstack{Est. total\\\\GFLOPs} \\\\',
        body[:-1],
        'Audio preprocessing is excluded. Core FLOPs equal twice MACs; non-core FLOPs are arithmetic estimates.')
    body = []
    FACTS['examples'] = {}
    for m, title in NAMES.items():
        b = BASE[m, 'lambda_0p1']
        candidates = [r for r in DATA[m,
            'lambda_0p1'] if r['family'] in ('recovery',
            'gap_affine') and r['recovery']['recovery_steps'] == 100]
        no_loss = [r for r in candidates if r['test']['correct'] >= b['test']['correct']]
        pool = no_loss or candidates
        best = min(pool,
            key=lambda r: ((-r['k'],) if no_loss else ()) + (-r['test']['correct'],
            r['parameters'],
            r['recovery']['learned_parameters'],
            tuple(r['drops']),
            r['case_id']))
        d = next((r for r in rows(m, 'lambda_0p1', 'F') if r['drops'] == best['drops']))
        saved = 100 * (1 - best['parameters'] / b['parameters'])
        FACTS['examples'][m] = dict(drops=best['drops'],
            method=method(best),
            scope=best['scope'],
            saved=saved,
            complete=b['accuracy_percent'],
            direct=d['accuracy_percent'],
            recovered=best['accuracy_percent'],
            delta=best['accuracy_change_pp'],
            trainable=best['recovery']['learned_parameters'],
            case_id=best['case_id'])
        label = 'Ins.\\ A' if best['family'] == 'gap_affine' else f"{best['mode']} ({best['scope']})"
        body.append(line([title,
            blocks(best['drops']),
            label,
            f'{saved:.2f}',
            f"{b['accuracy_percent']:.2f}",
            f"{best['accuracy_percent']:.2f}",
            f"${best['accuracy_change_pp']:+.2f}$"]))
    table('regularized_examples',
        'Compressed configurations identified in the test sweep of moment-regularized models.',
        'tab:regularized_examples',
        'lclrrrr',
        'Model & Deleted blocks & Recovery & \\shortstack{Params.\\\\saved (\\%)} & Complete (\\%) & Recovered (\\%) & $\\Delta$ (pp) \\\\',
        body,
        'Examples are selected retrospectively by test accuracy, prioritizing the largest loss-free deletion count, then accuracy; otherwise, the highest accuracy is used. Savings are relative to the complete model and include inserted parameters.')

def main():
    load()
    figures()
    tables()
    (PAPER / 'doc/summary.json').write_text(json.dumps(FACTS, indent=2), encoding='utf-8')
    print('Figures and tables written to', PAPER)
if __name__ == '__main__':
    main()
