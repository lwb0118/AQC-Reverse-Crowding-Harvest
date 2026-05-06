"""
ic_test.py
==========
Information Coefficient testing suite for AQC_Reverse factor.
Implements cross-sectional rank IC, ICIR, Newey-West robust t-statistic,
and IC decay analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_daily_rank_ic(panel, factor_col, return_col, min_stocks=10):
    """
    Compute daily cross-sectional rank IC.
    
    At each date t, rank-transform factor values and forward returns,
    then compute Pearson correlation between the ranks.
    """
    results = []
    for date, group in panel.groupby('date'):
        valid = group[[factor_col, return_col]].dropna()
        n = len(valid)
        if n < min_stocks:
            continue
        
        factor_ranks = stats.rankdata(valid[factor_col].values)
        return_ranks = stats.rankdata(valid[return_col].values)
        # Skip if no variance
        if np.std(factor_ranks) == 0 or np.std(return_ranks) == 0:
            continue
        ic, p_val = stats.pearsonr(factor_ranks, return_ranks)
        if np.isnan(ic):
            continue
        
        results.append({
            'date': date, 'ic': ic, 'p_value': p_val, 'n_stocks': n
        })
    
    if not results:
        raise ValueError("No valid IC dates found.")
    
    return pd.DataFrame(results).sort_values('date').reset_index(drop=True)


def compute_ic_statistics(ic_series):
    """
    Compute comprehensive IC statistics including Newey-West robust t-stat.
    """
    n = len(ic_series)
    ic_mean = float(np.mean(ic_series))
    ic_std = float(np.std(ic_series, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    
    # Simple t-test
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    
    # IC positive ratio
    ic_pos_ratio = float(np.sum(ic_series > 0) / n)
    
    # Newey-West robust t-statistic
    nw_tstat = _newey_west_tstat(ic_series)
    
    # Verdict
    if abs(nw_tstat) > 2.58:
        verdict = "*** Significant (99%)"
    elif abs(nw_tstat) > 1.96:
        verdict = "** Significant (95%)"
    elif abs(nw_tstat) > 1.645:
        verdict = "* Marginal (90%)"
    else:
        verdict = "Not significant"
    
    return {
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        't_stat': t_stat,
        'p_value': p_value,
        'ic_pos_ratio': ic_pos_ratio,
        'nw_tstat': nw_tstat,
        'n_days': n,
        'verdict': verdict,
    }


def full_ic_analysis(panel, factor_col='aqc_reverse',
                     return_cols=['fwd_return_1d', 'fwd_return_5d',
                                  'fwd_return_10d', 'fwd_return_20d'],
                     min_stocks=10):
    """
    Full IC analysis across multiple horizons.
    
    Returns dict of {horizon_label: {'ic_series': DataFrame, 'stats': dict}}
    """
    results = {}
    for rc in return_cols:
        label = rc.replace('fwd_return_', '').replace('d', '')
        try:
            ic_df = compute_daily_rank_ic(panel, factor_col, rc, min_stocks)
            stats_dict = compute_ic_statistics(ic_df['ic'].values)
            results[label] = {
                'ic_series': ic_df,
                'stats': stats_dict
            }
        except Exception as e:
            print(f"  [Skip] {rc}: {e}")
    
    return results


def full_ic_analysis_multi_target(panel, factor_col='aqc_reverse',
                                     targets=None, min_stocks=10):
    """
    Full IC analysis across multiple target types (return, drawdown, vol, reversal).

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [date, code, factor_col] and target columns.
    factor_col : str
    targets : dict of {label: list_of_cols}, optional
        e.g.: {
            'return': ['fwd_return_1d', 'fwd_return_5d', 'fwd_return_10d', 'fwd_return_20d'],
            'drawdown': ['fwd_drawdown_5d', 'fwd_drawdown_10d'],
            'vol': ['fwd_vol_5d', 'fwd_vol_10d'],
            'reversal': ['future_reversal_5w5'],
        }
        If None, uses default targets.
    min_stocks : int

    Returns
    -------
    dict of {target_type: {horizon_label: {'ic_series': ..., 'stats': ...}}}
    """
    if targets is None:
        targets = {
            'return': ['fwd_return_1d', 'fwd_return_5d',
                       'fwd_return_10d', 'fwd_return_20d'],
            'drawdown': ['fwd_drawdown_5d', 'fwd_drawdown_10d'],
            'vol': ['fwd_vol_5d', 'fwd_vol_10d'],
            'reversal': ['future_reversal_5w5'],
        }

    all_results = {}
    for target_type, cols in targets.items():
        available = [c for c in cols if c in panel.columns]
        if not available:
            print(f"  [Skip] {target_type}: no columns found in panel")
            continue

        try:
            results = full_ic_analysis(
                panel, factor_col=factor_col,
                return_cols=available, min_stocks=min_stocks
            )
            all_results[target_type] = results
        except Exception as e:
            print(f"  [Error] {target_type}: {e}")

    return all_results


def _newey_west_tstat(series):
    """
    Compute Newey-West HAC robust t-statistic.
    Adjusts for autocorrelation in the IC time series.
    """
    n = len(series)
    if n < 10:
        return 0.0
    
    ic_mean = np.mean(series)
    
    # Optimal lag: automatic bandwidth selection
    lag = int(np.ceil(4 * (n / 100) ** (2 / 9)))
    lag = min(lag, n - 2)
    
    # Compute autocovariance
    gamma_0 = np.var(series, ddof=0)
    
    newey_west_var = gamma_0
    for j in range(1, lag + 1):
        # Auto-covariance at lag j
        cov_j = np.mean((series[:-j] - ic_mean) * (series[j:] - ic_mean)) * (n / (n - 1))
        weight = 1 - j / (lag + 1)  # Bartlett kernel
        newey_west_var += 2 * weight * cov_j
    
    if newey_west_var <= 0:
        return 0.0
    
    nw_se = np.sqrt(newey_west_var / n)
    return ic_mean / nw_se if nw_se > 0 else 0.0
