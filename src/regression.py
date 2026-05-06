"""
regression.py
=============
Panel regression analysis for AQC_Reverse factor.
Tests:
1. AQC_Reverse → future returns
2. AQC → future drawdown
3. AIHeat × SmallIlliq → future reversal
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


def panel_regression(panel, y_col, x_col, controls=None, 
                      industry_fe=True, time_fe=True):
    """
    Run panel regression with industry and time fixed effects.
    
    Return(i,t+N) = α + β1·AQC_Reverse(i,t) + β2·Controls(i,t) + FE + ε
    
    Parameters
    ----------
    panel : pd.DataFrame
    y_col : str
        Dependent variable (e.g., 'fwd_return_5d').
    x_col : str
        Main independent variable (e.g., 'aqc_reverse').
    controls : list of str
        Control variable column names.
    industry_fe : bool
        Include industry fixed effects.
    time_fe : bool
        Include time fixed effects.
    
    Returns
    -------
    statsmodels regression result
    """
    df = panel.copy()
    
    # Prepare variables
    y = df[y_col].values
    X_cols = [x_col]
    
    if controls:
        X_cols += controls
    
    # Industry dummies
    if industry_fe and 'industry' in df.columns:
        industry_dummies = pd.get_dummies(df['industry'], prefix='ind',
                                           drop_first=True)
        df = pd.concat([df, industry_dummies], axis=1)
        X_cols += list(industry_dummies.columns)
    
    # Time dummies
    if time_fe:
        time_dummies = pd.get_dummies(df['date'].astype(str), 
                                       prefix='t', drop_first=True)
        df = pd.concat([df, time_dummies], axis=1)
        X_cols += list(time_dummies.columns)
    
    # Stack regressors
    X = df[X_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop rows with NaN
    valid = pd.concat([pd.Series(y, name='y'), X], axis=1).dropna()
    y_clean = valid['y'].values.astype(float)
    X_clean = valid.drop(columns=['y'])
    
    # Ensure all columns are numeric (convert dummies, dates, etc.)
    for col in X_clean.columns:
        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
    X_clean = X_clean.dropna(axis=1, how='all')
    
    X_clean = sm.add_constant(X_clean)
    
    if len(y_clean) < 50:
        raise ValueError(f"Too few observations: {len(y_clean)}")
    
    # Run OLS
    model = OLS(y_clean, X_clean.values.astype(float))
    results = model.fit()
    
    # Create a simplified result summary
    coef_idx = 0  # const is first
    main_idx = 1  # main x variable is second
    
    return {
        'coefficient': float(results.params[main_idx]),
        'std_error': float(results.bse[main_idx]),
        't_stat': float(results.tvalues[main_idx]),
        'p_value': float(results.pvalues[main_idx]),
        'r_squared': float(results.rsquared),
        'adj_r_squared': float(results.rsquared_adj),
        'n_obs': int(results.nobs),
        'n_vars': len(X_cols) + 1,
        'full_results': results,
    }


# ===========================================================================
# REGRESSION: AQC → FUTURE DRAWDOWN
# ===========================================================================

def regression_drawdown(panel, y_col='fwd_drawdown_5d', x_col='aqc',
                         controls=None, industry_fe=True, time_fe=True):
    """
    Test whether high AQC predicts future drawdown.

    Drawdown(i,t+N) = α + β·AQC(i,t) + β₂·Controls(i,t) + IndustryFE + TimeFE + ε

    If β > 0 and significant, high-AQC stocks experience larger
    future maximum drawdowns (crowding unwind risk).

    Parameters
    ----------
    panel : pd.DataFrame
    y_col : str
        Dependent variable (e.g., 'fwd_drawdown_5d').
    x_col : str
        Main independent variable (e.g., 'aqc' or 'aqc_zscore').
    controls : list of str, optional
        Control variable column names.
    industry_fe : bool
        Include industry fixed effects (requires 'industry' column).
    time_fe : bool
        Include time fixed effects.

    Returns
    -------
    dict with coefficient, std_error, t_stat, p_value, r_squared, etc.
    """
    if controls is None:
        controls = ['small_illiq', 'template_exposure', 'mom_20d',
                    'vol_20d', 'turnover_20d']

    return panel_regression(
        panel, y_col=y_col, x_col=x_col,
        controls=controls,
        industry_fe=industry_fe, time_fe=time_fe
    )


# ===========================================================================
# REGRESSION: INTERACTION EFFECTS (AIHeat × SmallIlliq × TemplateExposure)
# ===========================================================================

def regression_interaction(panel, y_col='fwd_return_5d',
                            heat_col='aiheat',
                            illiq_col='small_illiq',
                            template_col='template_exposure',
                            controls=None,
                            include_two_way=True,
                            industry_fe=True, time_fe=True):
    """
    Test interaction effects of AIHeat × SmallIlliq × TemplateExposure.

    Reversal(i,t+N) = α
                    + β₁·AIHeat×SmallIlliq×TemplateExposure
                    + β₂·Controls
                    + (β₃·AIHeat×SmallIlliq + β₄·AIHeat×TemplateExposure
                       + β₅·SmallIlliq×TemplateExposure)   [if include_two_way]
                    + FE + ε

    Parameters
    ----------
    panel : pd.DataFrame
    y_col : str
    heat_col : str
    illiq_col : str
    template_col : str
    controls : list of str, optional
    include_two_way : bool
        Include all pairwise interaction terms.
    industry_fe : bool
    time_fe : bool

    Returns
    -------
    dict keyed by interaction term name, with coefficient, t_stat, etc.
    """
    if controls is None:
        controls = ['mom_20d', 'vol_20d', 'turnover_20d', 'market_cap']

    df = panel.copy()

    # Ensure needed columns exist
    for col in [heat_col, illiq_col, template_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in panel.")

    # Create interaction terms
    df['interact_aqc'] = (df[heat_col] * df[illiq_col] * df[template_col]).fillna(0)

    # Prepare X columns
    X_cols = ['interact_aqc'] + controls

    if include_two_way:
        df['interact_heat_illiq'] = (df[heat_col] * df[illiq_col]).fillna(0)
        df['interact_heat_template'] = (df[heat_col] * df[template_col]).fillna(0)
        df['interact_illiq_template'] = (df[illiq_col] * df[template_col]).fillna(0)
        X_cols += ['interact_heat_illiq', 'interact_heat_template',
                   'interact_illiq_template']

    # Run regression using the existing machinery
    temp_y = df[y_col].values
    temp_X = df[X_cols].copy()
    for col in temp_X.columns:
        temp_X[col] = pd.to_numeric(temp_X[col], errors='coerce')

    valid = pd.concat([pd.Series(temp_y, name='y'), temp_X], axis=1).dropna()
    y_clean = valid['y'].values.astype(float)
    X_clean = valid.drop(columns=['y'])

    for col in X_clean.columns:
        X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
    X_clean = X_clean.dropna(axis=1, how='all')

    # Industry dummies
    if industry_fe and 'industry' in df.columns:
        ind_dummies = pd.get_dummies(df['industry'].loc[valid.index],
                                      prefix='ind', drop_first=True)
        for col in ind_dummies.columns:
            ind_dummies[col] = pd.to_numeric(ind_dummies[col], errors='coerce')
        X_clean = pd.concat([X_clean, ind_dummies], axis=1)

    # Time dummies
    if time_fe:
        t_dummies = pd.get_dummies(df['date'].astype(str).loc[valid.index],
                                    prefix='t', drop_first=True)
        for col in t_dummies.columns:
            t_dummies[col] = pd.to_numeric(t_dummies[col], errors='coerce')
        X_clean = pd.concat([X_clean, t_dummies], axis=1)

    # Drop any remaining NaN rows
    X_clean = X_clean.dropna()
    if isinstance(y_clean, np.ndarray):
        # numpy array: track indices manually
        y_clean = pd.Series(y_clean, index=X_clean.index)
    y_clean = y_clean.loc[X_clean.index]

    X_clean = sm.add_constant(X_clean)

    if len(y_clean) < 50:
        raise ValueError(f"Too few observations: {len(y_clean)}")

    model = OLS(y_clean.astype(float), X_clean.values.astype(float))
    results = model.fit()

    # Collect coefficient results for interaction term
    param_names = ['const'] + list(X_clean.columns[1:])
    main_idx = param_names.index('interact_aqc') if 'interact_aqc' in param_names else None

    interaction_results = {}
    if main_idx is not None:
        interaction_results['interact_aqc'] = {
            'coefficient': float(results.params.iloc[main_idx]),
            'std_error': float(results.bse.iloc[main_idx]),
            't_stat': float(results.tvalues.iloc[main_idx]),
            'p_value': float(results.pvalues.iloc[main_idx]),
        }

    # Also collect two-way terms if included
    if include_two_way:
        for term in ['interact_heat_illiq', 'interact_heat_template',
                      'interact_illiq_template']:
            if term in param_names:
                idx = param_names.index(term)
                interaction_results[term] = {
                    'coefficient': float(results.params.iloc[idx]),
                    'std_error': float(results.bse.iloc[idx]),
                    't_stat': float(results.tvalues.iloc[idx]),
                    'p_value': float(results.pvalues.iloc[idx]),
                }

    interaction_results['_model_summary'] = {
        'r_squared': float(results.rsquared),
        'adj_r_squared': float(results.rsquared_adj),
        'n_obs': int(results.nobs),
        'n_vars': len(param_names),
    }

    return interaction_results
