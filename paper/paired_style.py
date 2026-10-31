"""Build Section 5.5 direct-deletion and combined-recovery figures and table.

No model evaluation, training, manuscript compilation, or layout checks.
The reference and regularized checkpoints use identical deletion sets and
test manifests. Each plotted point is one deletion structure. Boxes show the
IQR and median, not uncertainty across independently trained checkpoints.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
PAPER = None
FIGURES = {'deletion': 'regularized_deletion', 'recovery': 'regularized_normalization_recovery'}
COLORS = {'F': '#707070',
    'S': '#0072B2',
    'A': '#D55E00',
    'S+A': '#009E73',
    'Interface A': '#8E5BA6',
    'Reference': '#707070',
    'Regularized': '#0072B2'}
MARKERS = {'F': 'o', 'S': 's', 'A': '^', 'S+A': 'D', 'Interface A': 's', 'Reference': 'o', 'Regularized': 's'}

def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans',
        'font.size': 7.5,
        'axes.titlesize': 7.5,
        'axes.labelsize': 7.5,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 7,
        'axes.linewidth': 0.65,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none'})

def distribution(ax, rows, x, color, marker, field, width):
    values = np.asarray([r[field] for r in rows], dtype=float)
    if len(rows) >= 10:
        ax.boxplot([values],
            positions=[x],
            widths=width,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            medianprops={'color': 'black',
            'lw': 0.8},
            whiskerprops={'color': color,
            'lw': 0.65},
            capprops={'color': color,
            'lw': 0.65},
            boxprops={'facecolor': color,
            'edgecolor': color,
            'alpha': 0.16,
            'lw': 0.65})
    elif len(rows) > 1:
        ax.plot([x - width / 2, x + width / 2], [np.median(values)] * 2, color='black', lw=0.8, zorder=4)
    jitter = np.asarray([int(hashlib.sha256(json.dumps(r['drops']).encode()).hexdigest()[:8],
        16) / 4294967295 - 0.5 for r in rows]) * width * 0.85
    if len(rows) == 1:
        jitter[:] = 0
    ax.scatter(x + jitter,
        values,
        s=7 if len(rows) > 50 else 11,
        marker=marker,
        c=color,
        alpha=0.5 if len(rows) > 50 else 0.85,
        edgecolors='none',
        zorder=3)

def plot_gain(groups, points, kind):
    """Horizontal paired differences, with explicit direction and mean labels."""
    style()
    selected = []
    summaries = []
    for group in groups:
        model = group['model']
        methods = ['S', 'A', 'S+A', 'Interface A'] if model in ('bc8', 'tc14') else ['A', 'Interface A']
        for k, n in enumerate(group['counts'], 1):
            for method in methods:
                variants = {v: {tuple(r['drops']): r for r in points if r['model'] == model and r['k'] == k and (r['method'] == method) and (r['variant'] == v)} for v in ('Reference',
                    'Regularized')}
                rr = []
                for drops, before in sorted(variants['Reference'].items()):
                    after = variants['Regularized'][drops]
                    rr.append(dict(model=model,
                        k=k,
                        method=method,
                        drops=list(drops),
                        reference_accuracy=before['accuracy'],
                        regularized_accuracy=after['accuracy'],
                        gain_pp=after['accuracy'] - before['accuracy'],
                        reference_case_id=before['case_id'],
                        regularized_case_id=after['case_id']))
                selected.extend(rr)
                vv = np.asarray([r['gain_pp'] for r in rr])
                summaries.append(dict(model=model,
                    k=k,
                    method=method,
                    n=n,
                    mean=statistics.mean(vv),
                    median=statistics.median(vv),
                    q25=float(np.quantile(vv,
                    0.25)),
                    q75=float(np.quantile(vv,
                    0.75)),
                    negative=int((vv < 0).sum()),
                    equal=int((vv == 0).sum()),
                    positive=int((vv > 0).sum())))
    values = np.array([r['gain_pp'] for r in selected])
    from matplotlib.ticker import MaxNLocator
    scales = {}
    for model in ('bc8', 'tc14', 'kwt1', 'kwmlp'):
        vv = [r['gain_pp'] for r in selected if r['model'] == model]
        lo = min(0, min(vv))
        hi = max(0, max(vv))
        margin = max((hi - lo) * 0.05, 0.05)
        ticks = MaxNLocator(nbins=4).tick_values(lo - margin, hi + margin)
        scales[model] = (float(ticks[0]), float(ticks[-1]), list(ticks))
    negative_color, positive_color, neutral_color = ('#D55E00', '#0072B2', '#777777')
    figure_height = 8.5
    fig, axes = plt.subplots(2,
        2,
        figsize=(138.6 / 25.4,
        figure_height),
        gridspec_kw={'height_ratios': [2,
        1]})
    fig.subplots_adjust(left=0.18, right=0.915, bottom=0.105, top=0.9, wspace=0.9, hspace=0.3)
    fig.text(0.5, 0.978, 'Accuracy difference = Regularized − Reference', ha='center', va='top', fontsize=8)
    for i, (ax, g) in enumerate(zip(axes.flat, groups)):
        low, high, ticks = scales[g['model']]
        model_values = [r['gain_pp'] for r in selected if r['model'] == g['model']]
        methods = ['S', 'A', 'S+A', 'Interface A'] if g['model'] in ('bc8', 'tc14') else ['A', 'Interface A']
        row_labels = []
        row_positions = []
        for k, n in enumerate(g['counts'], 1):
            start = (k - 1) * (len(methods) + 0.65)
            if k % 2 == 0:
                ax.axhspan(start - 0.4, start + len(methods) - 0.6, color='#F5F5F5', zorder=0)
            ax.text(-0.36,
                start + (len(methods) - 1) / 2,
                f'{k} ({n})',
                transform=ax.get_yaxis_transform(),
                fontsize=6.5,
                va='center',
                ha='center',
                color='#333333')
            for j, method in enumerate(methods):
                y = start + j
                rr = [r for r in selected if r['model'] == g['model'] and r['k'] == k and (r['method'] == method)]
                vv = np.asarray([r['gain_pp'] for r in rr])
                jitter = np.asarray([int(hashlib.sha256(json.dumps(r['drops']).encode()).hexdigest()[:8],
                    16) / 4294967295 - 0.5 for r in rr]) * 0.45
                if n == 1:
                    jitter[:] = 0
                colors = [positive_color if v > 0 else negative_color if v < 0 else neutral_color for v in vv]
                ax.scatter(vv,
                    y + jitter,
                    s=5 if n >= 10 else 11,
                    c=colors,
                    alpha=0.26 if n >= 10 else 0.8,
                    linewidths=0,
                    zorder=2)
                if n >= 10:
                    q25, q75 = np.quantile(vv, [0.25, 0.75])
                    ax.plot([q25, q75], [y, y], color='#222222', lw=2, solid_capstyle='butt', zorder=3)
                mean = float(np.mean(vv))
                ax.scatter([mean],
                    [y],
                    s=17,
                    marker='D',
                    facecolor='white',
                    edgecolor='#222222',
                    lw=0.7,
                    zorder=4)
                shown = 0.0 if abs(mean) < 0.005 else mean
                ax.text(1.045,
                    y,
                    f'{shown:+.2f}',
                    transform=ax.get_yaxis_transform(),
                    fontsize=6.5,
                    va='center',
                    ha='left',
                    color='#222222')
                row_labels.append('Ins. A' if method == 'Interface A' else method)
                row_positions.append(y)
        ax.set_ylim(row_positions[-1] + 0.65, -0.65)
        ax.set_yticks(row_positions, row_labels)
        ax.set_xlim(low, high)
        ax.set_xticks(ticks, [f'{v:+g}'.replace('-', '−') if v else '0' for v in ticks])
        ax.axvline(0, color='#555555', lw=0.8, ls='--', zorder=1)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='x', direction='out', length=2.5, width=0.65, pad=2, labelbottom=True)
        ax.tick_params(axis='y', length=0, pad=4, labelsize=6.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='#E7E7E7', lw=0.4)
        title = f"({chr(97 + i)}) {g['title']}"
        ax.annotate(title,
            (0,
            1),
            xycoords='axes fraction',
            xytext=(0,
            23),
            textcoords='offset points',
            fontsize=7.5,
            va='bottom')
        ax.annotate('k (n)',
            (-0.36,
            1),
            xycoords='axes fraction',
            xytext=(0,
            6),
            textcoords='offset points',
            fontsize=6.5,
            va='bottom',
            ha='center')
        ax.annotate('Mean',
            (1.045,
            1),
            xycoords='axes fraction',
            xytext=(0,
            6),
            textcoords='offset points',
            fontsize=6.5,
            ha='left',
            va='bottom')
    fig.text(0.5, 0.068, 'Paired accuracy difference (percentage points)', ha='center', fontsize=7.5)
    fig.text(0.47, 0.04, '← Reference higher', ha='right', color=negative_color, fontsize=7)
    fig.text(0.53, 0.04, 'Regularized higher →', ha='left', color=positive_color, fontsize=7)
    fig.legend(handles=[Line2D([],
        [],
        marker='o',
        color='#777777',
        ls='none',
        markersize=3,
        label='Deletion set'),
        Line2D([],
        [],
        marker='D',
        markerfacecolor='white',
        markeredgecolor='#222222',
        ls='none',
        markersize=4,
        label='Mean'),
        Line2D([],
        [],
        color='#222222',
        lw=2,
        label='IQR (n ≥ 10)')],
        loc='lower center',
        bbox_to_anchor=(0.5,
        0.005),
        ncol=3,
        frameon=False,
        fontsize=6.5)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(PAPER / 'fig' / f'{FIGURES[kind]}.{ext}', dpi=600, facecolor='white')
    plt.close(fig)
    payload = dict(figure=FIGURES[kind],
        kind=kind,
        points=selected,
        groups=summaries,
        plotted_points=len(selected),
        checkpoint_scope='Paired Reference and Regularized; both Retrained',
        statistic='Regularized minus Reference absolute accuracy for identical deletion sets and recovery settings, pp',
        x_limits_by_model={m: list(v[:2]) for m,
        v in scales.items()},
        value_axis='x',
        value_range=[float(values.min()),
        float(values.max())],
        display='Horizontal paired differences; all deletion sets; hollow diamond and numeric mean; IQR segment only for n>=10',
        direction='Negative: Reference higher; positive: Regularized higher',
        uncertainty='None; IQR describes deletion-set variation, not a confidence interval',
        baseline='zero paired difference',
        layout='2x2 horizontal point-range panels with grouped k (n) row labels',
        figure_size_mm=[138.6,
        figure_height * 25.4],
        layout_intent='Full-page portrait figure with expanded method-row spacing',
        scales='Model-specific linear axes, shared by BN/LN and inserted-affine recovery; all points retained',
        method_labels={'S': 'BN statistics',
        'A': 'BN/LN affine',
        'S+A': 'BN statistics and affine',
        'Interface A': 'Inserted affine (Ins. A)'},
        merged_legacy_figures=['Fig8: BN/LN recovery',
        'Fig9: inserted affine recovery'])
    (PAPER / 'doc' / f'{FIGURES[kind]}_points.json').write_text(json.dumps(payload,
        indent=2),
        encoding='utf-8')
    return {k: v for k, v in payload.items() if k not in ('points', 'groups')}

def plot(groups, points, kind):
    if kind != 'deletion':
        return plot_gain(groups, points, kind)
    style()
    direct = kind == 'deletion'
    height = 3.8475 if direct else 5.25 if kind == 'normalization' else 4.65
    fig, axes = plt.subplots(2, 2, figsize=(138.6 / 25.4, height), sharey=True)
    fig.subplots_adjust(left=0.12,
        right=0.985,
        bottom=0.16 if direct else 0.15,
        top=0.84 if direct else 0.865,
        wspace=0.18,
        hspace=0.8 if direct else 0.67)
    selected_points, group_counts = ([], [])
    for i, (ax, group) in enumerate(zip(axes.flat, groups)):
        model = group['model']
        ax.annotate(f"({chr(97 + i)}) {group['title']}",
            xy=(0,
            1),
            xycoords='axes fraction',
            xytext=(0,
            31),
            textcoords='offset points',
            fontsize=7.5,
            va='bottom')
        subtitle = f"Ref. ({group['reference_full']:.2f}%) | Reg. ({group['regularized_full']:.2f}%)" if direct else f"Regularized ({group['regularized_full']:.2f}%)"
        ax.annotate(subtitle,
            xy=(0,
            1),
            xycoords='axes fraction',
            xytext=(0,
            19),
            textcoords='offset points',
            fontsize=6.5,
            va='bottom')
        if direct:
            series = ['Reference', 'Regularized']
        elif kind == 'normalization':
            series = ['F', 'S', 'A', 'S+A'] if model in ('bc8', 'tc14') else ['F', 'A']
        else:
            series = ['F', 'A', 'Interface A']
        offsets = np.linspace(-0.3,
            0.3,
            4) if len(series) == 4 else np.linspace(-0.27,
            0.27,
            3) if len(series) == 3 else [-0.17,
            0.17]
        width = 0.16 if len(series) == 4 else 0.2 if len(series) == 3 else 0.26
        field = 'change_pp' if direct else 'accuracy'
        for k in range(1, 5):
            n = group['counts'][k - 1] if k <= len(group['counts']) else 0
            if not n:
                ax.text(k, -50 if direct else 45, 'N/A', ha='center', fontsize=7, color='#888888')
                continue
            label = f'n={n}' if n < 100 else f'n=\n{n}'
            ax.annotate(label,
                (k,
                1),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0,
                2),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=6.2,
                color='#444444')
            for name, offset in zip(series, offsets):
                variant, method = (name, 'F') if direct else ('Regularized', name)
                rows = [r for r in points if r['model'] == model and r['k'] == k and (r['variant'] == variant) and (r['method'] == method)]
                distribution(ax, rows, k + offset, COLORS[name], MARKERS[name], field, width)
                selected_points.extend(rows)
                group_counts.append({'model': model,
                    'k': k,
                    'variant': variant,
                    'method': method,
                    'n': n,
                    'mean': statistics.mean((r[field] for r in rows)),
                    'median': statistics.median((r[field] for r in rows))})
        ax.set_ylim(-100, 4) if direct else ax.set_ylim(0, 104)
        ax.set_yticks([-100, -75, -50, -25, 0] if direct else [0, 25, 50, 75, 100])
        ax.set_xlim(0.45, 4.55)
        ax.set_xticks([1, 2, 3, 4])
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=2.5, width=0.65, pad=2)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#DADADA', lw=0.45)
        ax.axhline(0 if direct else group['regularized_full'], color='#777777', lw=0.6, ls='--')
    fig.text(0.023,
        0.51,
        'Accuracy change, $\\Delta$ (pp)' if direct else 'Accuracy (%)',
        rotation=90,
        ha='center',
        va='center',
        fontsize=7.5)
    fig.text(0.55, 0.085, 'Number of deleted blocks', ha='center', fontsize=7.5)
    legend = ['Reference',
        'Regularized'] if direct else ['F',
        'S',
        'A',
        'S+A'] if kind == 'normalization' else ['F',
        'A',
        'Interface A']
    labels = {'A': 'A', 'Interface A': 'Ins. A'} if kind == 'affine' else {}
    fig.legend(handles=[Line2D([],
        [],
        color=COLORS[name],
        marker=MARKERS[name],
        ls='none',
        markersize=4,
        label=labels.get(name,
        name)) for name in legend],
        loc='lower center',
        bbox_to_anchor=(0.55,
        0.005),
        ncol=len(legend),
        frameon=False,
        fontsize=7.5)
    values = np.asarray([r[field] for r in selected_points])
    low, high = axes.flat[0].get_ylim()
    expected = {'deletion': 3510, 'normalization': 3848, 'affine': 5265}[kind]
    base = PAPER / 'fig' / FIGURES[kind]
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(base.with_suffix('.' + extension), dpi=600, facecolor='white')
    plt.close(fig)
    payload = {'figure': FIGURES[kind],
        'kind': kind,
        'points': selected_points,
        'groups': group_counts,
        'plotted_points': len(selected_points),
        'checkpoint_scope': 'Reference and Regularized' if direct else 'Regularized only',
        'y_limits': [low,
        high],
        'value_range': [float(values.min()),
        float(values.max())],
        'display': 'All points; IQR/median boxes for n>=10; median bars for 1<n<10',
        'whiskers': 'Most extreme observations within 1.5 IQR',
        'jitter': 'Deterministic, shared across methods for the same deleted positions',
        'baseline': 'zero change' if direct else 'Complete regularized-model accuracy'}
    payload['figure_size_mm'] = [138.6, height * 25.4]
    payload['panel_height_mm'] = float(axes.flat[0].get_position().height * height * 25.4)
    (PAPER / 'doc' / (FIGURES[kind] + '_points.json')).write_text(json.dumps(payload,
        ensure_ascii=False,
        indent=2),
        encoding='utf-8')
    return {k: v for k, v in payload.items() if k not in ('points', 'groups')}

def table(groups):
    lines = ['% Generated from saved paper results.',
        '\\begin{table}[htbp]',
        '\\caption{Complete-model accuracy and mean accuracy (\\%) after deletion',
        'and recovery for Reference and Regularized checkpoints.}',
        '\\label{tab:regularization_comparison}',
        '\\centering\\small',
        '\\setlength{\\tabcolsep}{4pt}',
        '\\begin{tabular}{lrrrrrrrr}',
        '\\toprule',
        ' & \\multicolumn{2}{c}{BC8 ($k=4$)} & \\multicolumn{2}{c}{TC14 ($k=3$)}',
        ' & \\multicolumn{2}{c}{KWT-1 ($k=4$)} & \\multicolumn{2}{c}{KW-MLP ($k=4$)} \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}',
        'Setting & Ref. & Reg. & Ref. & Reg. & Ref. & Reg. & Ref. & Reg. \\\\',
        '\\midrule']
    settings = [('Complete',
        'Full'),
        ('F',
        'F'),
        ('S',
        'S'),
        ('A',
        'A'),
        ('S+A',
        'S+A'),
        ('Ins.\\ A',
        'Interface A')]
    for label, method in settings:
        if method in ('S', 'Interface A'):
            lines.append('\\midrule')
        values = []
        for group in groups:
            for variant in ('Reference', 'Regularized'):
                if method == 'Full':
                    value = group['reference_full'] if variant == 'Reference' else group['regularized_full']
                else:
                    selected = [r['mean_accuracy'] for r in group['statistics'] if r['k'] == len(group['counts']) and r['variant'] == variant and (r['method'] == method)]
                    value = selected[0] if selected else None
                values.append(f'{value:.2f}' if value is not None else '---')
        lines.append(label + ' & ' + ' & '.join(values) + ' ' + '\\\\')
    lines.extend(['\\bottomrule',
        '\\end{tabular}',
        '\\par\\smallskip\\begin{minipage}{\\linewidth}\\footnotesize',
        'Ref./Reg. denote Reference/Regularized; means cover all compatible sets at the indicated deletion count. Dashes denote inapplicable settings.',
        '\\end{minipage}',
        '\\end{table}',
        ''])
    (PAPER / 'table' / 'regularization_comparison.tex').write_text('\n'.join(lines), encoding='utf-8')
