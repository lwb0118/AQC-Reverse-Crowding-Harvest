"""
visualization.py
================
Publication-quality visualizations for AQC_Reverse factor analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'primary': '#1a5276',
    'secondary': '#e74c3c',
    'accent': '#27ae60',
    'neutral': '#7f8c8d',
    'heat': ['#e74c3c', '#f39c12', '#27ae60'],
    'quintile': ['#2c3e50', '#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
}


def plot_ic_timeseries(ic_results, horizon='5', output_path=None):
    """Plot daily IC time series with cumulative overlay."""
    if horizon not in ic_results:
        return
    
    data = ic_results[horizon]
    ic_series = data['ic_series']
    stats = data['stats']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                     gridspec_kw={'height_ratios': [2, 1]})
    
    ics = ic_series['ic'].values
    
    # Daily IC
    ax1.bar(range(len(ics)), ics,
            color=[COLORS['accent'] if v > 0 else COLORS['secondary'] for v in ics],
            width=0.8, alpha=0.7, label='Daily IC')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=np.mean(ics), color=COLORS['primary'], linewidth=1.5,
                linestyle='--', label=f'Mean IC = {stats["ic_mean"]:.4f}')
    ax1.set_ylabel('Rank IC')
    ax1.set_title(f'Daily Rank IC — AQC_Reverse ({horizon}d Forward)')
    ax1.legend(loc='upper right')
    
    # Cumulative IC
    cum_ic = np.cumsum(ics)
    ax2.fill_between(range(len(cum_ic)), 0, cum_ic,
                      color=COLORS['primary'], alpha=0.3)
    ax2.plot(range(len(cum_ic)), cum_ic, color=COLORS['primary'],
             linewidth=1.5, label=f'Cumul. = {cum_ic[-1]:.2f}')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Cumulative IC')
    ax2.set_xlabel('Trading Day')
    ax2.legend(loc='upper right')
    
    fig.suptitle(f'ICIR={stats["icir"]:.3f} | NW t={stats["nw_tstat"]:.3f} | IC>0={stats["ic_pos_ratio"]:.1%}',
                 y=1.02, fontsize=11, fontstyle='italic', color=COLORS['neutral'])
    
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_quintile_returns(group_results, output_path=None):
    """Plot quintile portfolio returns."""
    means = group_results['group_means']
    groups = list(means.keys())
    values = [means[g] for g in groups]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(groups, values, color=COLORS['quintile'][:len(groups)],
                  width=0.6, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.0001 if val >= 0 else -0.0005),
                f'{val:.4%}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=11, fontweight='bold')
    
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Forward Return')
    ax.set_xlabel('AQC_Reverse Quintile (Q1=Low → Q5=High)')
    ax.set_title('Quintile Portfolio Returns — AQC_Reverse Factor')
    
    spread = group_results['spread']
    annualized = group_results['spread_annualized']
    mono = group_results['monotonicity']
    
    textstr = f'Q5−Q1: {spread:.4%} ({annualized:.1%} ann.)\nMonotonicity ρ = {mono:.3f}'
    ax.text(0.97, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_ic_decay(ic_results, output_path=None):
    """Plot IC decay across horizons."""
    if not ic_results:
        return
    
    horizons = sorted(ic_results.keys())
    icirs = [ic_results[h]['stats']['icir'] for h in horizons]
    nw_ts = [ic_results[h]['stats']['nw_tstat'] for h in horizons]
    x = [f'{h}d' for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors_icir = [COLORS['accent'] if v > 0.1 else COLORS['neutral']
                   if v > 0 else COLORS['secondary'] for v in icirs]
    ax1.bar(x, icirs, color=colors_icir, width=0.5, edgecolor='white')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=0.5, color=COLORS['accent'], linestyle='--', alpha=0.7, label='ICIR=0.5')
    ax1.set_ylabel('ICIR')
    ax1.set_title('ICIR Across Horizons')
    ax1.legend()
    
    colors_nw = [COLORS['accent'] if v > 1.96 else '#f39c12' if v > 1.28
                 else COLORS['secondary'] for v in nw_ts]
    ax2.bar(x, nw_ts, color=colors_nw, width=0.5, edgecolor='white')
    ax2.axhline(y=1.96, color=COLORS['primary'], linestyle='--', alpha=0.7, label='95% sig')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Newey-West t-stat')
    ax2.set_title('Statistical Significance')
    ax2.legend()
    
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()


def plot_aiheat_timeseries(aiheat, output_path=None):
    """Plot AIHeat time series with regime indicators."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(aiheat.index, aiheat.values, color=COLORS['primary'],
            linewidth=1.5, label='AIHeat')
    
    # High/low regime thresholds
    high_thresh = aiheat.quantile(0.7)
    low_thresh = aiheat.quantile(0.3)
    ax.axhline(y=high_thresh, color='red', linestyle='--', alpha=0.5,
               label=f'High regime (p70={high_thresh:.2f})')
    ax.axhline(y=low_thresh, color='green', linestyle='--', alpha=0.5,
               label=f'Low regime (p30={low_thresh:.2f})')
    
    # Highlight high AIHeat periods
    high_periods = aiheat > high_thresh
    ax.fill_between(aiheat.index, aiheat.values.min(), aiheat.values.max(),
                    where=high_periods.values, color='red', alpha=0.08)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('AIHeat (Z-score)')
    ax.set_title('AI Quant Tool Heat Index (AIHeat)')
    ax.legend()
    
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        print(f"  Saved: {output_path}")
    plt.close()


def plot_ml_feature_importance(importance_dict, title='Feature Importance',
                                output_path=None, top_n=10):
    """Plot ML feature importance."""
    if not importance_dict:
        return
    
    sorted_items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS['accent'] if v > 0 else COLORS['secondary'] for v in values]
    
    ax.barh(range(len(features)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()
    
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
    plt.close()
