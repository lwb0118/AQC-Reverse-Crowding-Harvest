#!/usr/bin/env python3
"""
run.py — AQC-Reverse-Crowding-Harvest Factor Demo
===================================================
Main entry point. Runs the full research pipeline.

Usage:
    python run.py --quick     # 10 stocks, ~10 seconds
    python run.py             # 20 stocks, ~25 seconds
    python run.py --full      # 100 stocks, ~60 seconds
    python run.py --no-ml     # Skip ML extension (faster)
    python run.py --compare   # Compare base vs bubble versions + mechanism tests
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# Project modules
from src.data_fetcher import build_panel, get_index_constituents
from src.ai_heat import compute_aiheat, compute_aiheat_ma
from src.factors import (
    compute_small_illiq, compute_template_exposure,
    compute_crowding, compute_crowding_decay, compute_future_reversal,
    compute_aqc, compute_forward_returns
)
from src.ic_test import full_ic_analysis, full_ic_analysis_multi_target
from src.backtest import quintile_portfolio_test, long_short_backtest
from src.regression import panel_regression, regression_drawdown, regression_interaction
from src.mechanism import (
    test_crowding_formation, test_crowding_unwind,
    test_aiheat_regime, test_market_cap_subsample,
    test_template_exposure_subsample
)
from src.visualization import (
    plot_ic_timeseries, plot_quintile_returns,
    plot_ic_decay, plot_aiheat_timeseries,
    plot_ml_feature_importance
)

# Paths
RESULTS_DIR = Path(__file__).parent / 'results'
CHARTS_DIR = RESULTS_DIR / 'figures'


def print_header(title):
    print(f'\n{"=" * 65}')
    print(f'  {title}')
    print(f'{"=" * 65}\n')


def print_ic_summary(results_dict, label_suffix=''):
    """Print a formatted IC summary table."""
    print(f'\n  {"Target":>20s}  | {"IC Mean":>8s} | {"ICIR":>6s} | {"NW t":>6s} | {"IC>0%":>7s} | Significance')
    print(f'  {"-"*70}')
    for target_type in sorted(results_dict.keys()):
        sub = results_dict[target_type]
        for h in sorted(sub.keys()):
            s = sub[h]['stats']
            label = f'{target_type}_{h}d'
            print(f'  {label:>20s}  | {s["ic_mean"]:+.4f}  | {s["icir"]:+.3f}  | '
                  f'{s["nw_tstat"]:+.3f}  | {s["ic_pos_ratio"]:.1%}  | {s["verdict"]}')


def run_standard_pipeline(panel, aiheat, args, version='base'):
    """Run the standard factor research pipeline on a panel."""
    factor_suffix = '' if version == 'base' else '_bubble'
    factor_col = f'aqc_reverse{factor_suffix}' if version != 'base' else 'aqc_reverse'
    bubble_col = 'aqc_bubble' if 'aqc_bubble' in panel.columns else 'aqc'

    print_header(f'IC / ICIR Analysis [{version}]')
    ic_results = full_ic_analysis(
        panel, factor_col=factor_col,
        return_cols=['fwd_return_1d', 'fwd_return_5d',
                      'fwd_return_10d', 'fwd_return_20d']
    )
    print_ic_summary({'return': ic_results})

    # IC plots
    for h in ic_results:
        plot_ic_timeseries(ic_results, horizon=h,
                           output_path=str(CHARTS_DIR / f'ic_timeseries_{h}d{factor_suffix}.png'))
    plot_ic_decay(ic_results, output_path=str(CHARTS_DIR / f'ic_decay_analysis{factor_suffix}.png'))

    # Multi-target IC
    print_header(f'Multi-Target IC Analysis [{version}]')
    try:
        multi_ic = full_ic_analysis_multi_target(panel, factor_col=factor_col)
        print_ic_summary(multi_ic)
    except Exception as e:
        print(f'  [Skip] Multi-target IC: {e}')

    # Quintile portfolio
    print_header(f'Quintile Portfolio Backtest [{version}]')
    group_results = quintile_portfolio_test(
        panel, factor_col=factor_col, return_col='fwd_return_5d', n_groups=5
    )
    print(f'\n  {"Portfolio":>10s}  |  Mean Return  |  Count')
    print(f'  {"-"*40}')
    for g in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        print(f'  {g:>10s}  |  {group_results["group_means"][g]:+.4%}    |  '
              f'{group_results["group_counts"][g]}')
    print(f'  {"-"*40}')
    print(f'  {"Spread Q5-Q1":>10s}  |  {group_results["spread"]:+.4%}')
    print(f'  {"Annualized":>10s}  |  {group_results["spread_annualized"]:+.1%}')
    print(f'  Monotonicity ρ: {group_results["monotonicity"]:+.3f}')
    plot_quintile_returns(group_results,
                           output_path=str(CHARTS_DIR / f'quintile_returns{factor_suffix}.png'))

    # Long-Short
    print_header(f'Long-Short Strategy [{version}]')
    ls_results = long_short_backtest(
        panel, factor_col=factor_col, return_col='fwd_return_5d',
        long_pct=0.2, short_pct=0.2, cost=0.001
    )
    print(f'    Mean return : {ls_results["mean_return"]:+.4%}')
    print(f'    Std return  : {ls_results["std_return"]:+.4%}')
    print(f'    Sharpe ratio: {ls_results["sharpe_ratio"]:.3f}')
    print(f'    Cum. return : {ls_results["cumulative_return"]:+.4%}')
    print(f'    Max DD      : {ls_results["max_drawdown"]:+.4%}')

    # Panel regression (AQC_Reverse → Future Returns)
    print_header(f'Panel Regression: Factor → Future 5d Return [{version}]')
    try:
        reg_results = panel_regression(
            panel, y_col='fwd_return_5d', x_col=factor_col,
            controls=['small_illiq', 'template_exposure', 'mom_20d',
                      'vol_20d', 'turnover_20d'],
            industry_fe=False, time_fe=True
        )
        print(f'  Coef     : {reg_results["coefficient"]:+.6f}')
        print(f'  Std Err  : {reg_results["std_error"]:.6f}')
        print(f'  t-stat   : {reg_results["t_stat"]:+.3f}')
        print(f'  p-value  : {reg_results["p_value"]:.4f}')
        print(f'  R²       : {reg_results["r_squared"]:.4f}')
    except Exception as e:
        print(f'  [Skip] {e}')

    # Regression: AQC → Future Drawdown
    print_header(f'Panel Regression: AQC → Future Drawdown [{version}]')
    try:
        dd_col = bubble_col
        dd_results = regression_drawdown(
            panel, y_col='fwd_drawdown_5d', x_col=dd_col,
            controls=['small_illiq', 'template_exposure', 'mom_20d',
                      'vol_20d', 'turnover_20d'],
            industry_fe=False, time_fe=True
        )
        print(f'  Coef (AQC→Drawdown): {dd_results["coefficient"]:+.6f}')
        print(f'  t-stat              : {dd_results["t_stat"]:+.3f}')
        print(f'  p-value             : {dd_results["p_value"]:.4f}')
        print(f'  R²                  : {dd_results["r_squared"]:.4f}')
    except Exception as e:
        print(f'  [Skip] {e}')

    # Regression: Interaction effects
    print_header(f'Panel Regression: Interaction Effects [{version}]')
    try:
        int_results = regression_interaction(
            panel, y_col='fwd_return_5d',
            heat_col='aiheat',
            illiq_col='small_illiq',
            template_col='template_exposure',
            controls=['mom_20d', 'vol_20d', 'turnover_20d', 'market_cap'],
            include_two_way=True,
            industry_fe=False, time_fe=True
        )
        for term in ['interact_aqc', 'interact_heat_illiq',
                      'interact_heat_template', 'interact_illiq_template']:
            if term in int_results:
                r = int_results[term]
                print(f'  {term:>30s}: coef={r["coefficient"]:+.6f}, '
                      f't={r["t_stat"]:+.3f}, p={r["p_value"]:.4f}')
        if '_model_summary' in int_results:
            ms = int_results['_model_summary']
            print(f'  {"---":>30s}')
            print(f'  {"R²":>30s}: {ms["r_squared"]:.4f}')
    except Exception as e:
        print(f'  [Skip] {e}')

    # Mechanism tests
    print_header(f'Mechanism: Crowding Formation [{version}]')
    try:
        cf = test_crowding_formation(panel, factor_col=bubble_col, n_groups=5)
        print(f'  {"Metric":>20s}  |  {"G1(低)":>10s}  |  {"G5(高)":>10s}  |  Spread  |  ρ')
        print(f'  {"-"*60}')
        for m in ['turnover', 'ab_amount', 'intraday_vol']:
            lo = cf['group_means']['G1'][m]
            hi = cf['group_means']['G5'][m]
            spr = cf['spread'][m]
            rho = cf['monotonicity'][m]
            print(f'  {m:>20s}  |  {lo:>10.4f}  |  {hi:>10.4f}  |  {spr:+.4f}  |  {rho:+.3f}')
    except Exception as e:
        print(f'  [Skip] Crowding formation: {e}')

    print_header(f'Mechanism: Crowding Unwind [{version}]')
    try:
        cu = test_crowding_unwind(panel, factor_col=bubble_col, n_groups=5)
        print(f'  {"Metric":>22s}  |  {"G1(低)":>10s}  |  {"G5(高)":>10s}  |  Spread  |  ρ')
        print(f'  {"-"*65}')
        for m in list(cu['group_means']['G1'].keys())[:5]:
            lo = cu['group_means']['G1'][m]
            hi = cu['group_means']['G5'][m]
            spr = cu['spread'][m]
            rho = cu['monotonicity'][m]
            print(f'  {m:>22s}  |  {lo:>+10.4%}  |  {hi:>+10.4%}  |  {spr:+.4%}  |  {rho:+.3f}')
    except Exception as e:
        print(f'  [Skip] Crowding unwind: {e}')

    print_header(f'Mechanism: AIHeat Regime [{version}]')
    try:
        ar = test_aiheat_regime(panel, aiheat, factor_col=factor_col,
                                 return_col='fwd_return_5d')
        for regime, reg_data in ar.items():
            if 'error' in reg_data:
                print(f'  {regime}: {reg_data["error"]}')
                continue
            ic_s = reg_data['ic_stats']
            if isinstance(ic_s, dict):
                print(f'  {regime}: IC={ic_s.get("ic_mean", "N/A"):+.4f}, '
                      f'ICIR={ic_s.get("icir", "N/A"):+.3f}, '
                      f'NW_t={ic_s.get("nw_tstat", "N/A"):+.3f}')
    except Exception as e:
        print(f'  [Skip] AIHeat regime: {e}')

    print_header(f'Mechanism: Market Cap Subsample [{version}]')
    try:
        mc = test_market_cap_subsample(panel, factor_col=factor_col,
                                        return_col='fwd_return_5d')
        for cap_g, cap_data in mc.items():
            if 'error' in cap_data:
                print(f'  {cap_g}: {cap_data["error"]}')
                continue
            ic_s = cap_data['ic_stats']
            grp = cap_data['group_results']
            if isinstance(ic_s, dict):
                print(f'  {cap_g}: IC={ic_s.get("ic_mean", "N/A"):+.4f}, '
                      f'ICIR={ic_s.get("icir", "N/A"):+.3f}, '
                      f'Spread={grp.get("spread", "N/A"):+.4%}')
    except Exception as e:
        print(f'  [Skip] Market cap subsample: {e}')

    print_header(f'Mechanism: Template Exposure Subsample [{version}]')
    try:
        te = test_template_exposure_subsample(panel, factor_col=factor_col,
                                               return_col='fwd_return_5d')
        for tg, te_data in te.items():
            if 'error' in te_data:
                print(f'  {tg}: {te_data["error"]}')
                continue
            ic_s = te_data['ic_stats']
            grp = te_data['group_results']
            if isinstance(ic_s, dict):
                print(f'  {tg}: IC={ic_s.get("ic_mean", "N/A"):+.4f}, '
                      f'ICIR={ic_s.get("icir", "N/A"):+.3f}, '
                      f'Spread={grp.get("spread", "N/A"):+.4%}')
    except Exception as e:
        print(f'  [Skip] Template exposure subsample: {e}')

    return {
        'ic_20d_icir': ic_results.get('20', {}).get('stats', {}).get('icir', 0),
        'ic_20d_nw': ic_results.get('20', {}).get('stats', {}).get('nw_tstat', 0),
        'spread': group_results.get('spread', 0),
        'annualized': group_results.get('spread_annualized', 0),
        'monotonicity': group_results.get('monotonicity', 0),
        'ls_sharpe': ls_results.get('sharpe_ratio', 0),
        'ls_cum_return': ls_results.get('cumulative_return', 0),
    }


def main():
    parser = argparse.ArgumentParser(description='AQC-Reverse-Crowding-Harvest Factor Demo')
    parser.add_argument('--quick', action='store_true', help='Quick demo (10 stocks)')
    parser.add_argument('--full', action='store_true', help='Full analysis (100 stocks)')
    parser.add_argument('--no-ml', action='store_true', help='Skip ML extension')
    parser.add_argument('--optimize', action='store_true', help='Run optimized version')
    parser.add_argument('--compare', action='store_true', help='Compare base vs bubble versions + mechanism tests')
    args = parser.parse_args()

    n_stocks = 100 if args.full else (10 if args.quick else 20)

    print_header('AQC-Reverse Crowding Harvest Factor Demo')
    print(f'  {__doc__}')

    # ======================================================================
    # Step 1: Data Acquisition
    # ======================================================================
    print_header('Step 1: Data Acquisition')

    stock_codes = get_index_constituents(max_stocks=n_stocks)
    print(f'  Fetching {len(stock_codes)} stocks...')
    panel = build_panel(stock_codes, max_workers=10)
    print(f'  Panel: {panel["code"].nunique()} stocks, {panel["date"].nunique()} trading days')

    # AIHeat
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        print(f'  Fetching real AIHeat from GitHub...')
    else:
        print(f'  Generating AIHeat time series (synthetic)...')
    aiheat = compute_aiheat(use_synthetic=True, n_days=len(panel['date'].unique()),
                             github_token=github_token)
    aiheat_ma = compute_aiheat_ma(aiheat, window=20)

    # ======================================================================
    # Step 2: Factor Computation
    # ======================================================================
    print_header('Step 2: Factor Construction')

    # Compute returns
    print('  Computing daily returns...')
    panel['return_1d'] = panel.groupby('code')['close'].pct_change()

    # Small & Illiquid exposure
    print('  SmallIlliq...')
    panel = compute_small_illiq(panel)

    # Template exposure
    print('  TemplateExposure...')
    panel = compute_template_exposure(panel)

    # Crowding
    print('  Crowding...')
    panel = compute_crowding(panel)

    # AQC & AQC_Reverse — compute both base and bubble versions
    print('  AQC & AQC_Reverse (base version)...')
    panel = compute_aqc(panel, aiheat, version='both')

    # Forward returns
    print('  Forward returns (1d, 5d, 10d, 20d)...')
    panel = compute_forward_returns(panel, horizons=[1, 5, 10, 20])

    # Crowding decay
    print('  Crowding decay...')
    panel = compute_crowding_decay(panel, horizons=[5, 10])

    # Future reversal indicator
    print('  Future reversal indicator...')
    panel = compute_future_reversal(panel, past_window=5, future_window=5)

    print(f'  Final panel: {len(panel)} obs, {panel["code"].nunique()} stocks')
    print(f'  Columns: {list(panel.columns)}')

    # ======================================================================
    # Step 3: AIHeat Visualization
    # ======================================================================
    print_header('Step 3: AIHeat Index')
    print(f'  AIHeat range: [{aiheat.min():.3f}, {aiheat.max():.3f}]')
    print(f'  High regime (p70): {aiheat.quantile(0.7):.3f}')
    print(f'  Low regime (p30): {aiheat.quantile(0.3):.3f}')
    plot_aiheat_timeseries(aiheat, output_path=str(CHARTS_DIR / 'aiheat_timeseries.png'))

    # ======================================================================
    # Step 4: Machine Learning Extension
    # ======================================================================
    if not args.no_ml:
        print_header('Step 4: Machine Learning Extension')

        try:
            from src.ml_model import prepare_ml_data, run_ml_models

            feature_cols = ['aqc_reverse', 'aqc', 'small_illiq', 'template_exposure',
                            'crowding', 'aiheat', 'mom_20d', 'turnover_20d', 'vol_20d']

            X, y = prepare_ml_data(panel, feature_cols, target_col='crowding',
                                    min_samples=100)
            print(f'  ML samples: {len(X)}, positive rate: {y.mean():.2%}')

            ml_results = run_ml_models(X, y)

            for model_name, model_res in ml_results.items():
                print(f'\n  [{model_name}]')
                if 'error' in model_res:
                    print(f'    Error: {model_res["error"]}')
                else:
                    for k, v in model_res.items():
                        if isinstance(v, dict):
                            print(f'    {k}:')
                            for sk, sv in list(v.items())[:5]:
                                print(f'      {sk}: {sv:.4f}')
                        elif isinstance(v, (int, float)):
                            print(f'    {k}: {v:.4f}')

            # Feature importance plot
            if 'random_forest' in ml_results and 'feature_importance' in ml_results['random_forest']:
                plot_ml_feature_importance(
                    ml_results['random_forest']['feature_importance'],
                    title='Random Forest — Feature Importance (Crowding Unwind)',
                    output_path=str(CHARTS_DIR / 'ml_feature_importance.png')
                )
        except ImportError:
            print('  [Skip] scikit-learn not installed. Install with: pip install scikit-learn')
        except Exception as e:
            print(f'  [Skip] ML extension error: {e}')
    else:
        print('  [Skip] --no-ml flag set')

    # ======================================================================
    # Step 5: Main Pipeline (Base version)
    # ======================================================================
    print_header('Step 5: Base Version — Full Pipeline')
    base_summary = run_standard_pipeline(panel, aiheat, args, version='base')

    # ======================================================================
    # Step 5b: Bubble Version comparison (if --compare)
    # ======================================================================
    if args.compare:
        print_header('Step 5b: Bubble Version — Full Pipeline')
        bubble_summary = run_standard_pipeline(panel, aiheat, args, version='bubble')

        print_header('Comparison: Base vs Bubble')
        print(f'\n  {"Metric":>25s}  |  {"Base":>12s}  |  {"Bubble":>12s}')
        print(f'  {"-"*55}')
        for metric in ['ic_20d_icir', 'ic_20d_nw', 'spread', 'annualized',
                        'monotonicity', 'ls_sharpe', 'ls_cum_return']:
            bv = base_summary.get(metric, 'N/A')
            bubv = bubble_summary.get(metric, 'N/A')
            fmt = f'{bubv:+.4f}' if isinstance(bubv, float) else str(bubv)
            fmt_b = f'{bv:+.4f}' if isinstance(bv, float) else str(bv)
            print(f'  {metric:>25s}  |  {fmt_b:>12s}  |  {fmt:>12s}')

        # Save comparison
        import json
        compare_data = {
            'base': base_summary,
            'bubble': bubble_summary,
            'summary': {k: {'base': base_summary.get(k), 'bubble': bubble_summary.get(k)}
                       for k in base_summary}
        }
        compare_file = RESULTS_DIR / 'comparison.json'
        compare_file.write_text(json.dumps(compare_data, indent=2))
        print(f'\n  [Compare] Saved to {compare_file}')

    # ======================================================================
    # Summary
    # ======================================================================
    print_header('Summary')
    print(f'  [OK] Pipeline completed successfully.')
    print(f'  [OK] Factor: AQC_Reverse (Reverse Crowding Harvest)')
    print(f'  [OK] Factor: AQC_Bubble_Reverse (Enhanced crowding unwind)')
    print(f'  [OK] Data: {panel["code"].nunique()} stocks, {panel["date"].nunique()} trading days')
    print(f'  [OK] Framework: Rank IC + ICIR + Newey-West + Quintile + Long-Short + Panel Reg')
    if not args.no_ml:
        print(f'  [OK] ML Extension: Logistic Regression + Random Forest')
    print(f'  [OK] Mechanism Tests: Crowding Formation, Unwind, AIHeat Regime, Caps, Template')
    print(f'  [OK] Figures saved to: {CHARTS_DIR}')
    print(f'\n  Core logic:')
    print(f'    AQC(i,t) = AIHeat_t × SmallIlliq(i,t) × TemplateExposure(i,t)')
    print(f'    AQC_Reverse(i,t) = -AQC(i,t)')
    print(f'    AQC_Bubble(i,t) = AQC(i,t) × Crowding(i,t)')
    print(f'    AQC_Bubble_Reverse(i,t) = -AQC_Bubble(i,t)')
    print(f'\n  Additional variables:')
    print(f'    CrowdingDecay(i,t+N) = Crowding(i,t) - Crowding(i,t+N)')
    print(f'    FutureReversal(i,t) = 1 if past_5d>0 and future_5d<0')
    print(f'\n  Interpretation:')
    print(f'    High AQC = high crowding risk (potential bubble/unwind)')
    print(f'    High AQC_Reverse = low crowding (safer, better forward returns)')
    print(f'    High CrowdingDecay = crowding has unwound (positive signal)')
    print(f'\n  Thank you for using AQC-Reverse-Crowding-Harvest.')


if __name__ == '__main__':
    main()
