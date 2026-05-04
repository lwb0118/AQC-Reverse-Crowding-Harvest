"""
factors.py
==========
Core factor construction for the AQC-Reverse framework:

  1. SmallIlliq       — Small-cap low-liquidity exposure
  2. TemplateExposure — Exposure to template/beginner strategies
  3. Crowding          — Current abnormal trading activity
  4. AQC               — AI-Quant Crowding exposure factor
  5. AQC_Reverse       — Reverse crowding harvest factor
"""

import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# SMALL & ILLIQUID EXPOSURE
# ===========================================================================

def compute_small_illiq(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SmallIlliq: small-cap + low-liquidity exposure.
    
    SmallIlliq(i,t) = AvgZ(-Size(i,t), -Amount(i,t), Illiq(i,t))
    
    Higher values = smaller, more illiquid stocks that are
    more vulnerable to localized capital flow.
    """
    df = panel.copy()
    
    # Amihud illiquidity: |return| / amount
    df['return_1d'] = df.groupby('code')['close'].pct_change()
    df['illiq'] = (df['return_1d'].abs() / (df['amount'] + 1)).fillna(0)
    if len(df) > 10:
        df['illiq'] = df['illiq'].clip(upper=df['illiq'].quantile(0.99))
    
    # Cross-sectional standardization per date
    illiq_list = []
    for d, grp in df.groupby('date'):
        g = grp.copy()
        n = len(g)
        if n < 5:
            continue
        
        # Size component (negative = small cap)
        size_z = _zscore(-np.log(g['market_cap'].values + 1))
        
        # Amount component (negative = low volume)
        amount_z = _zscore(-np.log(g['amount'].values + 1))
        
        # Illiquidity component
        illiq_z = _zscore(g['illiq'].values)
        
        g['small_illiq'] = (size_z + amount_z + illiq_z) / 3.0
        illiq_list.append(g)
    
    return pd.concat(illiq_list, ignore_index=True)


# ===========================================================================
# TEMPLATE STRATEGY EXPOSURE
# ===========================================================================

def compute_template_exposure(panel: pd.DataFrame, version='v1',
                                   mom_window=20, rev_window=1,
                                   turn_window=20, vol_window=20) -> pd.DataFrame:
    """
    Compute TemplateExposure: how likely a stock is selected by template strategies.
    
    TemplateExposure(i,t) = AvgZ(Mom, Rev, Turnover, Vol, -Price, -Size)
    
    Parameters
    ----------
    version : str
        'v1' = equal-weight all components (default)
        'v2' = weight by predictive power (size+turnover weighted more)
        'v3' = rank-based combination (more robust)
    """
    df = panel.copy()
    df.sort_values(['code', 'date'], inplace=True)
    
    # Momentum
    df['mom'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change(mom_window)
    )
    df = df.rename(columns={'mom': f'mom_{mom_window}d'})
    
    # Short-term reversal
    df['rev'] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change(rev_window)
    )
    df = df.rename(columns={'rev': f'rev_{rev_window}d'})
    
    # Turnover average
    turn_col = f'turnover_{turn_window}d'
    df[turn_col] = df.groupby('code')['turnover'].transform(
        lambda x: x.rolling(turn_window, min_periods=5).mean()
    )
    
    # Volatility
    vol_col = f'vol_{vol_window}d'
    df[vol_col] = df.groupby('code')['return_1d'].transform(
        lambda x: x.rolling(vol_window, min_periods=5).std()
    )
    
    # Cross-sectional standardization per date
    exp_list = []
    for d, grp in df.groupby('date'):
        g = grp.copy()
        n = len(g)
        if n < 5:
            continue
        
        if version == 'v3':
            # Rank-based combination (rank each component, then average ranks)
            from scipy import stats
            mom_r = stats.rankdata(g[f'mom_{mom_window}d'].fillna(0).values) / n
            rev_r = stats.rankdata(-g[f'rev_{rev_window}d'].fillna(0).values) / n
            turn_r = stats.rankdata(g[turn_col].fillna(0).values) / n
            vol_r = stats.rankdata(g[vol_col].fillna(0).values) / n
            price_r = stats.rankdata(-np.log(g['close'].values + 1)) / n
            size_r = stats.rankdata(-np.log(g['market_cap'].values + 1)) / n
            
            # Z-score the ranks
            comps = np.column_stack([mom_r, rev_r, turn_r, vol_r, price_r, size_r])
            comps = (comps - comps.mean(axis=0)) / (comps.std(axis=0, ddof=1) + 1e-8)
            
            g['template_exposure'] = comps.mean(axis=1)
        elif version == 'v2':
            # Weighted version: size and turnover get double weight
            mom_z = _zscore(g[f'mom_{mom_window}d'].fillna(0).values)
            rev_z = _zscore(-g[f'rev_{rev_window}d'].fillna(0).values)
            turn_z = _zscore(g[turn_col].fillna(0).values)
            vol_z = _zscore(g[vol_col].fillna(0).values)
            price_z = _zscore(-np.log(g['close'].values + 1))
            size_z = _zscore(-np.log(g['market_cap'].values + 1))
            
            # Size and turnover weighted 2x
            g['template_exposure'] = (mom_z + rev_z + 2*turn_z + vol_z + price_z + 2*size_z) / 8.0
        else:
            # v1: equal-weight
            mom_z = _zscore(g[f'mom_{mom_window}d'].fillna(0).values)
            rev_z = _zscore(-g[f'rev_{rev_window}d'].fillna(0).values)
            turn_z = _zscore(g[turn_col].fillna(0).values)
            vol_z = _zscore(g[vol_col].fillna(0).values)
            price_z = _zscore(-np.log(g['close'].values + 1))
            size_z = _zscore(-np.log(g['market_cap'].values + 1))
            
            g['template_exposure'] = (mom_z + rev_z + turn_z + vol_z + price_z + size_z) / 6.0
        
        exp_list.append(g)
    
    return pd.concat(exp_list, ignore_index=True)


# ===========================================================================
# CROWDING INDICATOR
# ===========================================================================

def compute_crowding(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Crowding: abnormal trading activity indicator.
    
    Crowding(i,t) = AvgZ(AbTurnover, AbAmount, IntradayVol, Illiq)
    
    Components:
    - AbTurnover: abnormal turnover vs 60-day mean
    - AbAmount: abnormal amount vs 60-day mean
    - IntradayVol: (high - low) / prev_close
    - Illiq: Amihud illiquidity
    """
    df = panel.copy()
    df.sort_values(['code', 'date'], inplace=True)
    
    # Pre-compute illiquidity
    df['return_1d'] = df.groupby('code')['close'].pct_change()
    df['illiq'] = (df['return_1d'].abs() / (df['amount'] + 1)).fillna(0)
    df['illiq'] = df['illiq'].clip(upper=df['illiq'].quantile(0.99))
    
    # Abnormal turnover
    df['turnover_ma60'] = df.groupby('code')['turnover'].transform(
        lambda x: x.rolling(60, min_periods=10).mean()
    )
    df['ab_turnover'] = df['turnover'] - df['turnover_ma60']
    
    # Abnormal amount (ratio to 60-day mean)
    df['amount_ma60'] = df.groupby('code')['amount'].transform(
        lambda x: x.rolling(60, min_periods=10).mean()
    )
    df['ab_amount'] = df['amount'] / (df['amount_ma60'] + 1)
    
    # Intraday volatility
    df['prev_close'] = df.groupby('code')['close'].shift(1)
    df['intraday_vol'] = (df['high'] - df['low']) / (df['prev_close'] + 1)
    df['intraday_vol'] = df['intraday_vol'].clip(upper=0.2)  # cap at 20%
    
    # Cross-sectional standardization
    crowd_list = []
    for d, grp in df.groupby('date'):
        g = grp.copy()
        n = len(g)
        if n < 5:
            continue
        
        ab_turn_z = _zscore(g['ab_turnover'].fillna(0).values)
        ab_amt_z = _zscore(g['ab_amount'].fillna(1).values)
        intra_z = _zscore(g['intraday_vol'].fillna(0).values)
        illiq_z = _zscore(g['illiq'].fillna(0).values)
        
        g['crowding'] = (ab_turn_z + ab_amt_z + intra_z + illiq_z) / 4.0
        crowd_list.append(g)
    
    return pd.concat(crowd_list, ignore_index=True)


# ===========================================================================
# AQC & AQC_REVERSE FACTORS
# ===========================================================================

def compute_aqc(panel: pd.DataFrame, aiheat_series: pd.Series,
                version='base', normalize='zscore',
                neutralize_industry=False) -> pd.DataFrame:
    """
    Compute AQC (AI-Quant Crowding), AQC_Bubble, and AQC_Reverse factors.
    
    AQC(i,t)      = AIHeat_t × SmallIlliq(i,t) × TemplateExposure(i,t)
    AQC_Bubble(i,t) = AQC(i,t) × Crowding(i,t)
    
    Parameters
    ----------
    version : str
        'base'   = only compute AQC (no crowing component)
        'bubble' = only compute AQC_Bubble (with crowding)
        'both'   = compute both AQC and AQC_Bubble
    normalize : str
        'zscore' = standard z-score
        'rank' = rank-based normalization (more robust)
    neutralize_industry : bool
        If True, orthogonalize AQC against industry dummies.
    """
    from scipy import stats as sp_stats
    
    df = panel.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Map AIHeat
    aiheat_df = pd.DataFrame({'date': aiheat_series.index, 'aiheat': aiheat_series.values})
    aiheat_df['date'] = pd.to_datetime(aiheat_df['date'].dt.date)
    df = df.merge(aiheat_df, on='date', how='left')
    df['aiheat'] = df['aiheat'].fillna(0)
    
    version = version.lower()
    compute_bubble = version in ('bubble', 'both')
    compute_base = version in ('base', 'both')
    
    # Compute base AQC
    if compute_base:
        df['aqc'] = df['aiheat'] * df['small_illiq'] * df['template_exposure']
    
    # Compute bubble version (includes current crowding)
    if compute_bubble and 'crowding' in df.columns:
        df['aqc_bubble'] = df['aiheat'] * df['small_illiq'] * df['template_exposure'] * df['crowding']
    elif compute_bubble:
        print('[Warning] Crowding column not found, cannot compute AQC_Bubble')
        df['aqc_bubble'] = df['aiheat'] * df['small_illiq'] * df['template_exposure']
    
    # Cross-sectional normalization
    aqc_list = []
    for d, grp in df.groupby('date'):
        g = grp.copy()
        
        if compute_base:
            vals = g['aqc'].values
            if len(vals) > 1:
                if normalize == 'rank':
                    ranks = sp_stats.rankdata(vals)
                    g['aqc_zscore'] = (ranks / len(ranks) - 0.5) * 2
                elif np.std(vals, ddof=1) > 0:
                    g['aqc_zscore'] = (vals - np.mean(vals)) / np.std(vals, ddof=1)
                else:
                    g['aqc_zscore'] = 0.0
            else:
                g['aqc_zscore'] = 0.0
            
            # Industry neutralization
            if neutralize_industry and 'industry' in g.columns:
                try:
                    import statsmodels.api as sm
                    y = g['aqc_zscore'].values
                    ind_dummies = pd.get_dummies(g['industry'], drop_first=True)
                    ind_dummies = ind_dummies.apply(pd.to_numeric, errors='coerce')
                    if len(ind_dummies.columns) > 0 and len(ind_dummies) > len(ind_dummies.columns) + 5:
                        X = sm.add_constant(ind_dummies.values.astype(float))
                        model = sm.OLS(y, X).fit()
                        residuals = model.resid
                        if np.std(residuals) > 0:
                            g['aqc_zscore'] = (residuals - np.mean(residuals)) / np.std(residuals)
                except:
                    pass
        
        if compute_bubble:
            vals_b = g['aqc_bubble'].values
            if len(vals_b) > 1:
                if normalize == 'rank':
                    ranks_b = sp_stats.rankdata(vals_b)
                    g['aqc_bubble_zscore'] = (ranks_b / len(ranks_b) - 0.5) * 2
                elif np.std(vals_b, ddof=1) > 0:
                    g['aqc_bubble_zscore'] = (vals_b - np.mean(vals_b)) / np.std(vals_b, ddof=1)
                else:
                    g['aqc_bubble_zscore'] = 0.0
            else:
                g['aqc_bubble_zscore'] = 0.0
        
        aqc_list.append(g)
    
    df = pd.concat(aqc_list, ignore_index=True)
    
    # AQC_Reverse = -AQC  (high AQC = crowded = bad, so reverse it)
    if compute_base and 'aqc_zscore' in df.columns:
        df['aqc_reverse'] = -df['aqc_zscore']
    
    # AQC_Bubble_Reverse = -AQC_Bubble
    if compute_bubble and 'aqc_bubble_zscore' in df.columns:
        df['aqc_bubble_reverse'] = -df['aqc_bubble_zscore']
    
    return df


# ===========================================================================
# CROWDING DECAY (Future crowding relative to current)
# ===========================================================================

def compute_crowding_decay(panel: pd.DataFrame, horizons=[1, 5, 10]) -> pd.DataFrame:
    """
    Compute CrowdingDecay: future crowding relative to current crowding.

    CrowdingDecay(i,t+N) = Crowding(i,t) - Crowding(i,t+N)

    A positive value means crowding has decreased (unwind),
    a negative value means crowding has increased further.

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain 'crowding' column.
    horizons : list of int
        Forward horizons to compute decay.

    Returns
    -------
    pd.DataFrame with added columns: crowding_decay_{h}d
    """
    df = panel.copy()
    df.sort_values(['code', 'date'], inplace=True)

    for h in horizons:
        col = f'crowding_decay_{h}d'
        # Future crowding
        future_crowding = df.groupby('code')['crowding'].transform(
            lambda x: x.shift(-h)
        )
        # Decay = current - future (positive means crowding decreased)
        df[col] = df['crowding'] - future_crowding

    return df


# ===========================================================================
# FUTURE REVERSAL INDICATOR
# ===========================================================================

def compute_future_reversal(panel: pd.DataFrame,
                             past_window=5, future_window=5) -> pd.DataFrame:
    """
    Compute FutureReversal: 1 if stock rose recently and will fall soon.

    FutureReversal(i,t) = 1 if PastReturn_5d > 0 AND FutureReturn_5d < 0
                          0 otherwise

    This captures the "bubble then pop" pattern:
    high crowding drives up price (past 5d >0), then it reverses (future 5d <0).

    Parameters
    ----------
    panel : pd.DataFrame
        Must contain 'close' column.
    past_window : int
        Window for past return.
    future_window : int
        Window for future return.

    Returns
    -------
    pd.DataFrame with added column: future_reversal_{past}w{future}
    """
    df = panel.copy()
    df.sort_values(['code', 'date'], inplace=True)

    # Past return
    past_col = f'past_return_{past_window}d'
    df[past_col] = df.groupby('code')['close'].transform(
        lambda x: x.pct_change(past_window)
    )

    # Future return
    future_col = f'fwd_return_{future_window}d'
    if future_col not in df.columns:
        df[future_col] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-future_window) / x - 1
        )

    # Reversal indicator
    rev_col = f'future_reversal_{past_window}w{future_window}'
    df[rev_col] = ((df[past_col] > 0) & (df[future_col] < 0)).astype(int)

    return df


def compute_forward_returns(panel: pd.DataFrame, horizons=[1, 5, 10, 20]) -> pd.DataFrame:
    """
    Compute forward returns, drawdowns, and volatility for specified horizons.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Must contain columns: [date, code, close]
    horizons : list of int
        Forward horizons in trading days.
    
    Returns
    -------
    pd.DataFrame with added columns for each horizon.
    """
    df = panel.copy()
    df.sort_values(['code', 'date'], inplace=True)
    
    for h in horizons:
        col = f'fwd_return_{h}d'
        df[col] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-h) / x - 1
        )
        
        # Maximum drawdown over the horizon
        dd_col = f'fwd_drawdown_{h}d'
        max_col = f'fwd_high_{h}d'
        df[max_col] = df.groupby('code')['close'].transform(
            lambda x: x.rolling(h, min_periods=1).max().shift(-h)
        )
        df[dd_col] = (df['close'] - df[max_col]) / df['close']
        
        # Forward volatility
        vol_col = f'fwd_vol_{h}d'
        mp = max(2, h)
        df[vol_col] = df.groupby('code')['return_1d'].transform(
            lambda x, hh=h: x.shift(-1).rolling(hh, min_periods=hh).std()
        )
    
    return df


def _zscore(arr):
    """Z-score normalization of a 1D array."""
    if len(arr) < 2:
        return np.zeros_like(arr)
    if np.std(arr, ddof=1) > 0:
        return (arr - np.mean(arr)) / np.std(arr, ddof=1)
    return np.zeros_like(arr)
