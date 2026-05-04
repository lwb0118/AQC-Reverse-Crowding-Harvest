#!/usr/bin/env python3
"""
permutation_test.py
===================
Permutation (random shuffle) test to check for overfitting.
Shuffles the factor values and re-runs IC analysis many times.
If the real IC is better than 95% of shuffled ICs, the signal is likely real.
"""

import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_fetcher import build_panel, get_index_constituents
from src.ai_heat import compute_aiheat
from src.factors import (
    compute_small_illiq, compute_template_exposure,
    compute_crowding, compute_aqc, compute_forward_returns
)
from src.ic_test import compute_daily_rank_ic, compute_ic_statistics

N_STOCKS = 100
N_PERMUTATIONS = 100
verbose = True


def run_one_permutation(panel, factor_col, return_col, min_stocks=10):
    """Run IC analysis for one permutation."""
    ic_df = compute_daily_rank_ic(panel, factor_col, return_col, min_stocks)
    stats = compute_ic_statistics(ic_df['ic'].values)
    return stats['ic_mean'], stats['icir'], stats['nw_tstat']


def main():
    print('=' * 60)
    print('  Permutation Test for AQC_Reverse Factor')
    print(f'  {N_PERMUTATIONS} shuffles, {N_STOCKS} stocks')
    print('=' * 60)
    
    # Build data (same as run.py)
    print('\n[1/3] Building data...')
    stock_codes = get_index_constituents(max_stocks=N_STOCKS)
    panel = build_panel(stock_codes, max_workers=10, verbose=False)
    
    # AIHeat (try GitHub first, fallback synthetic)
    git_token = os.environ.get('GITHUB_TOKEN')
    aiheat = compute_aiheat(use_synthetic=True,
                             n_days=len(panel['date'].unique()),
                             github_token=git_token)
    
    # Build factors
    print('[2/3] Building factors...')
    p = panel.copy()
    p['return_1d'] = p.groupby('code')['close'].pct_change()
    p = compute_small_illiq(p)
    p = compute_template_exposure(p)
    p = compute_crowding(p)
    p = compute_aqc(p, aiheat, use_bubble=False)
    p = compute_forward_returns(p, horizons=[5, 20])
    
    # Run real IC
    print('[3/3] Running permutation test (100x)...')
    
    horizons = ['5', '20']
    results = {}
    
    for h in horizons:
        rc = f'fwd_return_{h}d'
        
        # Real IC
        real_ic = run_one_permutation(p, 'aqc_reverse', rc)
        
        # Permutation ICs
        perm_ic_means = []
        perm_icirs = []
        perm_nw_ts = []
        
        for i in range(N_PERMUTATIONS):
            # Shuffle the factor column across all dates
            p_shuff = p.copy()
            p_shuff['aqc_reverse'] = p_shuff.groupby('date')['aqc_reverse'].transform(
                lambda x: x.sample(frac=1, random_state=i).values
            )
            
            perm_ic = run_one_permutation(p_shuff, 'aqc_reverse', rc)
            perm_ic_means.append(perm_ic[0])
            perm_icirs.append(perm_ic[1])
            perm_nw_ts.append(perm_ic[2])
            
            if verbose and (i+1) % 20 == 0:
                print(f'  {h}d: {i+1}/{N_PERMUTATIONS} permutations done')
        
        # Results
        real_mean, real_icir, real_nw = real_ic
        
        perm_mean_mean = np.mean(perm_ic_means)
        perm_mean_std = np.std(perm_ic_means, ddof=1)
        
        # Percentile rank of real result
        pct_mean = np.mean([1 if x < real_mean else 0 for x in perm_ic_means])
        pct_nw = np.mean([1 if x < real_nw else 0 for x in perm_nw_ts])
        
        results[h] = {
            'real_ic_mean': real_mean,
            'real_icir': real_icir,
            'real_nw_t': real_nw,
            'perm_ic_mean_mean': perm_mean_mean,
            'perm_ic_mean_std': perm_mean_std,
            'pctile_real_ic': pct_mean,
            'pctile_real_nw': pct_nw,
            'perm_ic_means': perm_ic_means,
            'perm_nw_ts': perm_nw_ts,
        }
    
    # Print results
    print('\n' + '=' * 60)
    print('  PERMUTATION TEST RESULTS')
    print('=' * 60)
    
    for h in horizons:
        r = results[h]
        print(f'\n  --- {h}-day horizon ---')
        print(f'  IC Mean: Real={r["real_ic_mean"]:+.4f}  '
              f'Permutation={r["perm_ic_mean_mean"]:+.4f}'
              f' (sd={r["perm_ic_mean_std"]:.4f})')
        print(f'  ICIR:    Real={r["real_icir"]:+.3f}')
        print(f'  NW t:    Real={r["real_nw_t"]:+.3f}')
        print(f'  Real IC beats {r["pctile_real_ic"]:.0%} of shuffled ICs')
        print(f'  Real NW t beats {r["pctile_real_nw"]:.0%} of shuffled NW ts')
        
        if r['pctile_real_nw'] >= 0.95:
            verdict = 'SIGNAL IS REAL (95% confidence)'
        elif r['pctile_real_nw'] >= 0.90:
            verdict = 'Signal marginal (90% confidence)'
        else:
            verdict = 'Signal may be noise — caution advised'
        
        print(f'  ==> {verdict}')
    
    # Summary
    print('\n' + '=' * 60)
    print('  Summary:')
    print(f'    Best horizon: {max(results, key=lambda h: results[h]["pctile_real_nw"])}d')
    print(f'    Permutations: {N_PERMUTATIONS}')
    print(f'    Stocks used: {N_STOCKS}')
    print('=' * 60)


if __name__ == '__main__':
    main()
