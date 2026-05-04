"""
backtest.py
===========
Quintile portfolio backtest and long-short strategy for AQC_Reverse factor.
"""

import numpy as np
import pandas as pd
from scipy import stats


def quintile_portfolio_test(panel, factor_col='aqc_reverse',
                             return_col='fwd_return_5d', n_groups=5):
    """
    Sort stocks into quintile portfolios by factor value,
    then compute mean forward return per group.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [date, code, factor_col, return_col]
    factor_col : str
    return_col : str
    n_groups : int
    
    Returns
    -------
    dict with keys: group_means, group_counts, spread, spread_annualized, monotonicity
    """
    results = []
    
    for date, group in panel.groupby('date'):
        valid = group[[factor_col, return_col]].dropna()
        if len(valid) < n_groups * 2:
            continue
        
        # Assign to groups
        valid['group'] = pd.qcut(valid[factor_col].rank(method='first'),
                                  q=n_groups, labels=[f'Q{i+1}' for i in range(n_groups)])
        results.append(valid)
    
    if not results:
        raise ValueError("No valid test dates found.")
    
    all_data = pd.concat(results, ignore_index=True)
    
    # Group means
    group_means = {}
    group_counts = {}
    for g in [f'Q{i+1}' for i in range(n_groups)]:
        grp = all_data[all_data['group'] == g]
        group_means[g] = float(grp[return_col].mean())
        group_counts[g] = len(grp)
    
    # Spread: highest - lowest
    spread = group_means[f'Q{n_groups}'] - group_means['Q1']
    
    # Annualized spread (assuming 250 trading days / 5 = 50 periods)
    spread_annualized = spread * 50
    
    # Monotonicity test: Spearman correlation between group rank and return
    group_order = [f'Q{i+1}' for i in range(n_groups)]
    mean_returns = [group_means[g] for g in group_order]
    rho, _ = stats.spearmanr(range(n_groups), mean_returns)
    
    return {
        'group_means': group_means,
        'group_counts': group_counts,
        'spread': spread,
        'spread_annualized': spread_annualized,
        'monotonicity': rho,
    }


def long_short_backtest(panel, factor_col='aqc_reverse',
                         return_col='fwd_return_5d',
                         long_pct=0.2, short_pct=0.2,
                         cost=0.001):
    """
    Long-short strategy: long top 20%, short bottom 20%.
    
    Parameters
    ----------
    panel : pd.DataFrame
    factor_col : str
    return_col : str
    long_pct : float
        Fraction of stocks to go long (top).
    short_pct : float
        Fraction of stocks to go short (bottom).
    cost : float
        One-way transaction cost ratio.
    
    Returns
    -------
    dict with time series and performance stats.
    """
    period_returns = []
    weights_list = []
    
    for date, group in panel.groupby('date'):
        valid = group[[factor_col, return_col, 'code']].dropna()
        if len(valid) < 5:
            continue
        
        n_total = len(valid)
        n_long = max(1, min(int(n_total * long_pct), n_total // 2 - 1))
        n_short = max(1, min(int(n_total * short_pct), n_total // 2 - 1))
        
        # Rank by factor descending
        valid = valid.sort_values(factor_col, ascending=False)
        long_stocks = valid.head(n_long)
        short_stocks = valid.tail(n_short)
        
        # Equal weight
        long_return = long_stocks[return_col].mean()
        short_return = short_stocks[return_col].mean()
        
        # Net return (long - short) minus transaction costs
        net_return = (long_return - short_return) - cost * 2
        
        period_returns.append({
            'date': date,
            'long_return': long_return,
            'short_return': short_return,
            'net_return': net_return,
        })
    
    if not period_returns:
        raise ValueError("No valid backtest periods.")
    
    perf = pd.DataFrame(period_returns)
    
    # Performance statistics
    mean_net = perf['net_return'].mean()
    std_net = perf['net_return'].std()
    sharpe = mean_net / std_net * np.sqrt(52) if std_net > 0 else 0  # weekly
    cumulative = (1 + perf['net_return']).cumprod()
    max_dd = (cumulative / cumulative.cummax() - 1).min()
    
    return {
        'performance': perf,
        'mean_return': mean_net,
        'std_return': std_net,
        'sharpe_ratio': sharpe,
        'cumulative_return': float(cumulative.iloc[-1]) - 1 if len(cumulative) > 0 else 0,
        'max_drawdown': max_dd,
        'n_periods': len(perf),
    }
