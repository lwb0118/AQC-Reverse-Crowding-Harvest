# AQC-Reverse-Crowding-Harvest
# AI-Quant Crowding Reverse Harvest Factor Research

from .factors import (
    compute_small_illiq,
    compute_template_exposure,
    compute_crowding,
    compute_crowding_decay,
    compute_future_reversal,
    compute_aqc,
    compute_forward_returns,
)
from .ic_test import (
    compute_daily_rank_ic,
    compute_ic_statistics,
    full_ic_analysis,
    full_ic_analysis_multi_target,
)
from .regression import (
    panel_regression,
    regression_drawdown,
    regression_interaction,
)
from .backtest import (
    quintile_portfolio_test,
    long_short_backtest,
)
from .mechanism import (
    test_crowding_formation,
    test_crowding_unwind,
    test_aiheat_regime,
    test_market_cap_subsample,
    test_template_exposure_subsample,
)
