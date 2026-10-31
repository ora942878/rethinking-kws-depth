"""Figures 2--5 from complete four-model runs; no manuscript compilation.

Figure contract: every eligible assignment/deletion is one observation.
Reuse changes target parameters, not execution depth. Recovery compares the
same deletion sets at a fixed 100-step budget. LayerNorm has no S/S+A modes.
All points are shown; boxes are IQR/median, whiskers 1.5 IQR; n<10 uses points
and a median bar when n>1. No confidence intervals or significance claims.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
MODELS = [('bc8',
    'BC-ResNet-8',
    'Reference'),
    ('tc14',
    'TC-ResNet-14',
    'Reference'),
    ('kwt1',
    'KWT-1',
    'Reference'),
    ('kwmlp',
    'KW-MLP',
    'Reference')]
NAMES = {2: 'single_block_reuse', 3: 'multi_block_deletion', 4: 'multi_block_reuse', 5: 'exhaustive_recovery'}
DROP = {'bc8': [8, 28, 56, 70], 'tc14': [3, 3, 1, 0], 'kwt1': [12, 66, 220, 495], 'kwmlp': [12, 66, 220, 495]}
REUSE = {'bc8': [8,
    18,
    8,
    1],
    'tc14': [0,
    0,
    0,
    0],
    'kwt1': [22,
    190,
    816,
    1820],
    'kwmlp': [22,
    190,
    816,
    1820]}
COLORS = {'F': '#707070', 'S': '#0072B2', 'A': '#D55E00', 'S+A': '#009E73'}
MARKERS = {'F': 'o', 'S': 's', 'A': '^', 'S+A': 'D'}

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

def header(ax, i, title, checkpoint, baseline, multi=False):
    if multi:
        ax.annotate(f'({chr(97 + i)}) {title}',
            xy=(0,
            1),
            xycoords='axes fraction',
            xytext=(0,
            31),
            textcoords='offset points',
            fontsize=7.5,
            va='bottom')
        ax.annotate(f'{checkpoint} ({baseline:.2f}%)',
            xy=(0,
            1),
            xycoords='axes fraction',
            xytext=(0,
            19),
            textcoords='offset points',
            fontsize=6.5,
            va='bottom')
    else:
        ax.set_title(f'({chr(97 + i)}) {title}', loc='left', y=1.22, pad=0)
        ax.text(0, 1.14, f'{checkpoint} ({baseline:.2f}%)', transform=ax.transAxes, fontsize=6.5, va='top')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=2.5, width=0.65, pad=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#DADADA', lw=0.45)

def distribution(ax, rows, x, color, marker, yfield, width=0.45):
    if not rows:
        return
    values = np.asarray([r[yfield] for r in rows])
    if len(rows) >= 10:
        box = ax.boxplot([values],
            positions=[x],
            widths=width,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            medianprops=dict(color='black',
            lw=0.8),
            whiskerprops=dict(color=color,
            lw=0.65),
            capprops=dict(color=color,
            lw=0.65),
            boxprops=dict(facecolor=color,
            edgecolor=color,
            alpha=0.16,
            lw=0.65))
    elif len(rows) > 1:
        ax.plot([x - width / 2, x + width / 2], [np.median(values)] * 2, color='black', lw=0.8, zorder=4)
    jitter = np.array([int(r['case_id'][:8],
        16) / 4294967295 - 0.5 for r in rows]) * width * 0.85 if len(rows) > 1 else np.zeros(1)
    ax.scatter(x + jitter,
        values,
        s=7 if len(rows) > 50 else 11,
        marker=marker,
        c=color,
        alpha=0.5 if len(rows) > 50 else 0.85,
        edgecolors='none',
        zorder=3)
PANEL_HEIGHT = 4.05 * (0.84 - 0.16) / (2 + 0.8)

def horizontal_figure(width_ratios=None):
    height = PANEL_HEIGHT + 0.56 + 0.6
    fig, axes = plt.subplots(1,
        4,
        figsize=(138.6 / 25.4,
        height),
        sharey=True,
        gridspec_kw={'width_ratios': width_ratios} if width_ratios else None)
    fig.subplots_adjust(left=0.105,
        right=0.987,
        bottom=0.56 / height,
        top=(0.56 + PANEL_HEIGHT) / height,
        wspace=0.23)
    return (fig, axes)

def plot_single(groups, sources):
    fig, axes = horizontal_figure([1.05, 0.78, 1.15, 1.15])
    values = [r['accuracy_change_pp'] for rows in groups.values() for r in rows]
    step = 10 if min(values) >= -40 else 20
    bottom = min(-step, int(np.floor(min(values) / step)) * step)
    top = max(1, int(np.ceil(max(values))))
    for i, (ax, (kind, title, checkpoint)) in enumerate(zip(axes, MODELS)):
        header(ax, i, title, checkpoint, sources[kind]['baseline_accuracy'])
        rows = [r for r in groups[kind] if r['family'] == 'reuse']
        for left, color, marker, offset in [(True, '#0072B2', '^', -0.1), (False, '#C67900', 's', 0.1)]:
            selected = [r for r in rows if (r['copies'][0][0] < r['copies'][0][1]) == left]
            ax.scatter([r['copies'][0][1] + offset for r in selected],
                [r['accuracy_change_pp'] for r in selected],
                s=20,
                c=color,
                marker=marker,
                linewidths=0.25,
                edgecolors='white',
                zorder=3)
        positions = sorted({r['copies'][0][1] for r in rows})
        if positions:
            ax.set_xticks(positions)
            ax.set_xlim(min(positions) - 0.6, max(positions) + 0.6)
            if len(positions) > 8:
                plt.setp(ax.get_xticklabels(), rotation=50, ha='right', rotation_mode='anchor')
        else:
            ax.set_xticks([])
            ax.text(0.5,
                0.48,
                'No compatible\nadjacent pair',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=6.5,
                color='#666666')
        ax.set_ylim(bottom, top)
        ax.set_yticks(np.arange(bottom, 1, step))
        ax.axhline(0, color='#A0A0A0', lw=0.65)
        ax.set_xlabel('Block position', labelpad=3)
    axes[0].set_ylabel('Accuracy change, $\\Delta$ (pp)')
    fig.legend(handles=[Line2D([],
        [],
        marker=m,
        color=c,
        linestyle='none',
        markersize=4,
        label=l) for m,
        c,
        l in [('^',
        '#0072B2',
        'Left neighbor'),
        ('s',
        '#C67900',
        'Right neighbor')]],
        loc='lower center',
        bbox_to_anchor=(0.55,
        -0.005),
        ncol=2,
        frameon=False,
        fontsize=7)
    return fig

def plot_multi(figure, groups, sources):
    recovery = figure == 5
    if recovery:
        fig, axes = plt.subplots(2, 2, figsize=(138.6 / 25.4, 5.25), sharey=True)
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.15, top=0.865, wspace=0.18, hspace=0.67)
    else:
        fig, axes = horizontal_figure()
    for i, (ax, (kind, title, checkpoint)) in enumerate(zip(axes.flat, MODELS)):
        header(ax, i, title, checkpoint, sources[kind]['baseline_accuracy'], multi=True)
        baseline = sources[kind]['baseline_accuracy']
        modes = ['F',
            'S',
            'A',
            'S+A'] if recovery and kind in ['bc8',
            'tc14'] else ['F',
            'A'] if recovery else ['F']
        offsets = np.linspace(-0.3, 0.3, len(modes)) if len(modes) > 1 else [0]
        counts = REUSE[kind] if figure == 4 else DROP[kind]
        if figure == 4 and (not any(counts)):
            ax.text(0.5,
                0.48,
                'No compatible\nadjacent pair',
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=6.5,
                color='#666666')
        for k, n in enumerate(counts, 1):
            if n == 0:
                if figure != 4:
                    ax.text(k, 45 if recovery else -50, 'N/A', ha='center', fontsize=7, color='#888888')
                continue
            label = f'n={n}' if n < 100 else f'n=\n{n}'
            ax.annotate(label,
                xy=(k,
                1),
                xycoords=ax.get_xaxis_transform(),
                xytext=(0,
                2),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=6.2,
                color='#444444' if recovery else '#0072B2' if figure == 3 else '#CF5701')
            for mode, offset in zip(modes, offsets):
                selected = [r for r in groups[kind] if r['family'] != 'baseline' and len(r['drops'] or r['copies']) == k and (r['mode'] == mode)]
                distribution(ax,
                    selected,
                    k + offset,
                    COLORS[mode] if recovery else '#0072B2' if figure == 3 else '#CF5701',
                    MARKERS[mode] if recovery else 'o',
                    'accuracy_percent' if recovery else 'accuracy_change_pp',
                    0.16 if len(modes) == 4 else 0.24 if recovery else 0.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlim(0.45, 4.55)
        ax.set_ylim(0, 104) if recovery else ax.set_ylim(-100, 4)
        ax.set_yticks([0, 25, 50, 75, 100] if recovery else [-100, -75, -50, -25, 0])
        ax.axhline(baseline if recovery else 0, color='#777777', lw=0.6, ls='--')
    fig.text(0.023,
        0.51,
        'Accuracy (%)' if recovery else 'Accuracy change, $\\Delta$ (pp)',
        rotation=90,
        ha='center',
        va='center')
    fig.text(0.55,
        0.085 if recovery else 0.045,
        'Number of reused target positions' if figure == 4 else 'Number of deleted blocks',
        ha='center')
    if recovery:
        fig.legend(handles=[Line2D([],
            [],
            marker=MARKERS[m],
            color=COLORS[m],
            linestyle='none',
            markersize=4,
            label=m) for m in COLORS],
            loc='lower center',
            bbox_to_anchor=(0.55,
            0.005),
            ncol=4,
            frameon=False)
    return fig
