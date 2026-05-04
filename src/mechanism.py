"""
mechanism.py
============
Mechanism tests for the AQC_Reverse framework.

Provides functions to test the economic mechanisms underlying
the crowding-unwind hypothesis:

  1. test_crowding_formation   — Is high-AQC associated with high turnover,
                                 high abnormal volume, high intraday vol?
  2. test_crowding_unwind       — Does high-AQC subsequently experience
                                 turnover decline, volume drop, price reversal?
  3. test_aiheat_regime         — Does AQC_Reverse work better under high AIHeat?
  4. test_market_cap_subsample  — Does AQC_Reverse work better in small caps?
  5. test_template_exposure_subsample — Does it work better in high template stocks?
"""

import numpy as np
import pandas as pd


def _get_ic_func(panel, factor_col, return_col, min_stocks=10):
    """
    Internal: compute rank IC for a given factor and return column.
    Returns dict with ic_mean, icir, nw_tstat, etc.
    """
    from src.ic_test import compute_daily_rank_ic, compute_ic_statistics
    ic_df = compute_daily_rank_ic(panel, factor_col, return_col, min_stocks)
    return compute_ic_statistics(ic_df['ic'].values)


# ===========================================================================
# 1. CROWDING FORMATION MECHANISM
# ===========================================================================

def test_crowding_formation(panel, factor_col='aqc', n_groups=5):
    """
    Test crowding formation mechanism.

    Hypothesis: High-AQC stocks have higher turnover, abnormal amount,
    and intraday volatility during the formation period.

    For each date, stocks are sorted by factor_col into n_groups.
    Within each group, we compute the mean of crowding-related metrics:
      - turnover (current raw turnover)
      - ab_amount (abnormal amount ratio)
      - intraday_vol ((high - low) / prev_close)

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [date, code, factor_col, turnover, ab_amount, intraday_vol]
    factor_col : str
        Column used to sort stocks (e.g., 'aqc').
    n_groups : int
        Number of groups.

    Returns
    -------
    dict with:
      - 'group_means': {group: {metric: value}} — time-series average per group
      - 'group_means_avg': {metric: {group: value}} — same transposed
      - 'spread': {metric: high_group - low_group}
      - 'monotonicity': {metric: spearman rho}
    """
    from scipy import stats

    metrics = ['turnover', 'ab_amount', 'intraday_vol']
    # Ensure needed columns exist
    missing = [m for m in metrics if m not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing required columns: {missing}")

    group_labels = [f'G{i+1}' for i in range(n_groups)]
    all_results = []  # list of {date, group, metric, value}

    for date, grp in panel.groupby('date'):
        valid = grp[[factor_col] + metrics].dropna()
        if len(valid) < n_groups * 2:
            continue

        valid['group'] = pd.qcut(valid[factor_col].rank(method='first'),
                                  q=n_groups,
                                  labels=group_labels)

        for g in group_labels:
            sub = valid[valid['group'] == g]
            for m in metrics:
                all_results.append({
                    'date': date, 'group': g, 'metric': m,
                    'value': sub[m].mean()
                })

    if not all_results:
        raise ValueError("No valid dates for crowding formation test.")

    result_df = pd.DataFrame(all_results)

    # Time-series average per group per metric
    group_means = {}
    for g in group_labels:
        group_means[g] = {}
        for m in metrics:
            vals = result_df[(result_df['group'] == g) &
                             (result_df['metric'] == m)]['value']
            group_means[g][m] = float(vals.mean())

    # Transpose view
    group_means_avg = {}
    for m in metrics:
        group_means_avg[m] = {}
        for g in group_labels:
            group_means_avg[m][g] = group_means[g][m]

    # Spread (highest group - lowest group)
    spread = {}
    for m in metrics:
        spread[m] = group_means[group_labels[-1]][m] - group_means[group_labels[0]][m]

    # Monotonicity
    monotonicity = {}
    for m in metrics:
        rank_order = list(range(n_groups))
        mean_vals = [group_means[g][m] for g in group_labels]
        rho, _ = stats.spearmanr(rank_order, mean_vals)
        monotonicity[m] = rho

    return {
        'group_means': group_means,
        'group_means_avg': group_means_avg,
        'spread': spread,
        'monotonicity': monotonicity,
    }


# ===========================================================================
# 2. CROWDING UNWIND MECHANISM
# ===========================================================================

def test_crowding_unwind(panel, factor_col='aqc', n_groups=5):
    """
    Test crowding unwind mechanism.

    Hypothesis: High-AQC stocks subsequently experience:
      - Turnover decline (negative future turnover change)
      - Amount decline (negative or lower forward abnormal amount)
      - Price reversal (negative forward returns / drawdown)

    For each date, stocks are sorted by factor_col into n_groups.
    Within each group, we compute the mean of future unwind metrics:
      - future turnover change (forward turnover minus current)
      - future amount change (forward ab_amount ratio)
      - fwd_return_5d (forward return)
      - fwd_drawdown_5d (forward max drawdown)

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [date, code, factor_col] and relevant future metrics.
    factor_col : str
    n_groups : int

    Returns
    -------
    dict with group_means, spread, monotonicity for each unwind metric.
    """
    from scipy import stats

    # Determine which future metrics are available
    candidate_metrics = [
        'fwd_return_5d', 'fwd_return_1d', 'fwd_return_10d',
        'fwd_drawdown_5d', 'fwd_drawdown_10d',
        'crowding_decay_5d', 'crowding_decay_10d',
    ]
    metrics = [m for m in candidate_metrics if m in panel.columns]
    if not metrics:
        # Fallback: use only forward returns
        metrics = [c for c in panel.columns
                   if c.startswith('fwd_return_') or
                   c.startswith('fwd_drawdown_')][:3]

    group_labels = [f'G{i+1}' for i in range(n_groups)]
    all_results = []

    for date, grp in panel.groupby('date'):
        valid = grp[[factor_col] + metrics].dropna()
        if len(valid) < n_groups * 2:
            continue

        valid['group'] = pd.qcut(valid[factor_col].rank(method='first'),
                                  q=n_groups,
                                  labels=group_labels)

        for g in group_labels:
            sub = valid[valid['group'] == g]
            for m in metrics:
                all_results.append({
                    'date': date, 'group': g, 'metric': m,
                    'value': sub[m].mean()
                })

    if not all_results:
        raise ValueError("No valid dates for crowding unwind test.")

    result_df = pd.DataFrame(all_results)

    group_means = {}
    for g in group_labels:
        group_means[g] = {}
        for m in metrics:
            vals = result_df[(result_df['group'] == g) &
                             (result_df['metric'] == m)]['value']
            group_means[g][m] = float(vals.mean())

    group_means_avg = {}
    for m in metrics:
        group_means_avg[m] = {}
        for g in group_labels:
            group_means_avg[m][g] = group_means[g][m]

    spread = {}
    for m in metrics:
        spread[m] = group_means[group_labels[-1]][m] - group_means[group_labels[0]][m]

    monotonicity = {}
    for m in metrics:
        rank_order = list(range(n_groups))
        mean_vals = [group_means[g][m] for g in group_labels]
        rho, _ = stats.spearmanr(rank_order, mean_vals)
        monotonicity[m] = rho

    return {
        'group_means': group_means,
        'group_means_avg': group_means_avg,
        'spread': spread,
        'monotonicity': monotonicity,
    }


# ===========================================================================
# 3. AIHEAT REGIME ANALYSIS
# ===========================================================================

def test_aiheat_regime(panel, aiheat, factor_col='aqc_reverse',
                        return_col='fwd_return_5d', n_terciles=3,
                        ic_test_func=None):
    """
    Test AQC_Reverse performance across AIHeat regimes.

    Split sample into low/medium/high AIHeat terciles,
    then compute IC statistics and quintile returns in each regime.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [date, code, factor_col, return_col].
    aiheat : pd.Series
        AIHeat time series (DatetimeIndex).
    factor_col : str
    return_col : str
    n_terciles : int
        Number of AIHeat regimes (default 3: low/med/high).
    ic_test_func : callable, optional
        If not provided, uses internal _get_ic_func.

    Returns
    -------
    dict of {regime_label: {'ic_stats': ..., 'group_means': ...}}
    """
    from src.ic_test import compute_daily_rank_ic, compute_ic_statistics
    from src.backtest import quintile_portfolio_test

    if ic_test_func is None:
        ic_test_func = _get_ic_func

    # Assign AIHeat regime to panel
    df = panel.copy()
    # Use the provided aiheat series (could also be already in panel)
    if 'aiheat' not in df.columns:
        aiheat_df = pd.DataFrame({
            'date': aiheat.index,
            'aiheat': aiheat.values
        })
        aiheat_df['date'] = pd.to_datetime(aiheat_df['date'].dt.date)
        df = df.merge(aiheat_df, on='date', how='left')
    df['aiheat'] = df['aiheat'].fillna(0)

    # Compute regime thresholds
    thresh_low = aiheat.quantile(1.0 / n_terciles)
    thresh_high = aiheat.quantile((n_terciles - 1.0) / n_terciles)

    # Regime labels
    if n_terciles == 3:
        labels = ['Low_AIHeat', 'Mid_AIHeat', 'High_AIHeat']
    else:
        labels = [f'Regime_{i+1}' for i in range(n_terciles)]

    results = {}
    for i, reg_label in enumerate(labels):
        if n_terciles == 3:
            if i == 0:
                mask = df['aiheat'] <= thresh_low
            elif i == 1:
                mask = (df['aiheat'] > thresh_low) & (df['aiheat'] <= thresh_high)
            else:
                mask = df['aiheat'] > thresh_high
        else:
            q_low = i / n_terciles
            q_high = (i + 1.0) / n_terciles
            lo = aiheat.quantile(q_low)
            hi = aiheat.quantile(q_high)
            if i == 0:
                mask = df['aiheat'] <= hi
            elif i == n_terciles - 1:
                mask = df['aiheat'] > lo
            else:
                mask = (df['aiheat'] > lo) & (df['aiheat'] <= hi)

        sub_panel = df[mask].copy()
        if len(sub_panel) < 50:
            results[reg_label] = {'error': f'Too few samples ({len(sub_panel)})'}
            continue

        # IC statistics
        try:
            ic_stats = ic_test_func(sub_panel, factor_col, return_col)
        except Exception as e:
            ic_stats = {'error': str(e)}

        # Quintile portfolio
        try:
            group_res = quintile_portfolio_test(
                sub_panel, factor_col=factor_col,
                return_col=return_col, n_groups=5
            )
        except Exception as e:
            group_res = {'error': str(e)}

        results[reg_label] = {
            'ic_stats': ic_stats,
            'group_results': group_res,
            'n_dates': sub_panel['date'].nunique(),
        }

    return results


# ===========================================================================
# 4. MARKET CAP SUBSAMPLE ANALYSIS
# ===========================================================================

def test_market_cap_subsample(panel, factor_col='aqc_reverse',
                               return_col='fwd_return_5d',
                               ic_test_func=None):
    """
    Test AQC_Reverse across market cap subsamples.

    Each date, stocks are split into Small/Mid/Large terciles by market_cap,
    then IC statistics and quintile returns are computed within each tercile.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [date, code, factor_col, return_col, market_cap].
    factor_col : str
    return_col : str
    ic_test_func : callable, optional

    Returns
    -------
    dict of {cap_group: {'ic_stats': ..., 'group_results': ...}}
    """
    from src.ic_test import compute_daily_rank_ic, compute_ic_statistics
    from src.backtest import quintile_portfolio_test

    if ic_test_func is None:
        ic_test_func = _get_ic_func

    if 'market_cap' not in panel.columns:
        raise ValueError("Panel must contain 'market_cap' column.")

    df = panel.copy()
    cap_groups = ['Small_Cap', 'Mid_Cap', 'Large_Cap']
    results = {}

    for cap_label in cap_groups:
        sub_list = []
        for date, grp in df.groupby('date'):
            valid = grp[[factor_col, return_col, 'market_cap', 'date', 'code']].dropna()
            if len(valid) < 9:
                continue
            # Tercile split by market cap
            lo = valid['market_cap'].quantile(1 / 3)
            hi = valid['market_cap'].quantile(2 / 3)
            if cap_label == 'Small_Cap':
                mask = valid['market_cap'] <= lo
            elif cap_label == 'Mid_Cap':
                mask = (valid['market_cap'] > lo) & (valid['market_cap'] <= hi)
            else:
                mask = valid['market_cap'] > hi
            sub_list.append(valid[mask])

        if not sub_list:
            results[cap_label] = {'error': 'No data'}
            continue

        sub_panel = pd.concat(sub_list, ignore_index=True)
        if len(sub_panel) < 50:
            results[cap_label] = {'error': f'Too few samples ({len(sub_panel)})'}
            continue

        try:
            ic_stats = ic_test_func(sub_panel, factor_col, return_col)
        except Exception as e:
            ic_stats = {'error': str(e)}

        try:
            group_res = quintile_portfolio_test(
                sub_panel, factor_col=factor_col,
                return_col=return_col, n_groups=5
            )
        except Exception as e:
            group_res = {'error': str(e)}

        results[cap_label] = {
            'ic_stats': ic_stats,
            'group_results': group_res,
            'n_dates': sub_panel['date'].nunique(),
            'n_stocks': sub_panel['code'].nunique(),
        }

    return results


# ===========================================================================
# 5. TEMPLATE EXPOSURE SUBSAMPLE ANALYSIS
# ===========================================================================

def test_template_exposure_subsample(panel, factor_col='aqc_reverse',
                                      return_col='fwd_return_5d',
                                      ic_test_func=None):
    """
    Test AQC_Reverse across template exposure subsamples.

    Each date, stocks are split into Low/Mid/High terciles by
    template_exposure, then IC statistics and quintile returns
    are computed within each tercile.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain [date, code, factor_col, return_col, template_exposure].
    factor_col : str
    return_col : str
    ic_test_func : callable, optional

    Returns
    -------
    dict of {template_group: {'ic_stats': ..., 'group_results': ...}}
    """
    from src.ic_test import compute_daily_rank_ic, compute_ic_statistics
    from src.backtest import quintile_portfolio_test

    if ic_test_func is None:
        ic_test_func = _get_ic_func

    if 'template_exposure' not in panel.columns:
        raise ValueError("Panel must contain 'template_exposure' column.")

    df = panel.copy()
    template_groups = ['Low_Template', 'Mid_Template', 'High_Template']
    results = {}

    for tg_label in template_groups:
        sub_list = []
        for date, grp in df.groupby('date'):
            valid = grp[[factor_col, return_col, 'template_exposure', 'date', 'code']].dropna()
            if len(valid) < 9:
                continue
            lo = valid['template_exposure'].quantile(1 / 3)
            hi = valid['template_exposure'].quantile(2 / 3)
            if tg_label == 'Low_Template':
                mask = valid['template_exposure'] <= lo
            elif tg_label == 'Mid_Template':
                mask = (valid['template_exposure'] > lo) & (valid['template_exposure'] <= hi)
            else:
                mask = valid['template_exposure'] > hi
            sub_list.append(valid[mask])

        if not sub_list:
            results[tg_label] = {'error': 'No data'}
            continue

        sub_panel = pd.concat(sub_list, ignore_index=True)
        if len(sub_panel) < 50:
            results[tg_label] = {'error': f'Too few samples ({len(sub_panel)})'}
            continue

        try:
            ic_stats = ic_test_func(sub_panel, factor_col, return_col)
        except Exception as e:
            ic_stats = {'error': str(e)}

        try:
            group_res = quintile_portfolio_test(
                sub_panel, factor_col=factor_col,
                return_col=return_col, n_groups=5
            )
        except Exception as e:
            group_res = {'error': str(e)}

        results[tg_label] = {
            'ic_stats': ic_stats,
            'group_results': group_res,
            'n_dates': sub_panel['date'].nunique(),
            'n_stocks': sub_panel['code'].nunique(),
        }

    return results
