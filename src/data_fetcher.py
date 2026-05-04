"""
data_fetcher.py
===============
Fetch A-share stock data and AIHeat proxy data via akshare.
Used for academic demonstration only — not for live trading.
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Default stock sample (diversified across sectors)
DEFAULT_STOCKS = [
    "600519", "000858", "600036", "000333", "000651",  # 消费/金融龙头
    "002415", "300750", "600276", "002304", "000568",  # 科技/医药/白酒
    "000001", "000002", "601166", "600900", "601318",  # 金融/公用
    "002714", "000625", "601012", "300124", "002475",  # 制造/新能源
]


def get_index_constituents(index_code="000300", max_stocks=None):
    """
    Fetch index constituent list (default: CSI 300).
    Returns a list of 6-digit stock codes.
    """
    try:
        df = ak.index_stock_cons(symbol=index_code)
        code_col = None
        for col in ['constituent_code', '品种代码', df.columns[0]]:
            if col in df.columns:
                code_col = col
                break
        if code_col is None:
            return DEFAULT_STOCKS[:max_stocks] if max_stocks else DEFAULT_STOCKS
        
        codes = [str(c).strip().zfill(6) for c in df[code_col].tolist()]
        codes = [c for c in codes if c.isdigit() and len(c) == 6]
        codes = sorted(set(codes))
        
        if max_stocks and max_stocks < len(codes):
            codes = codes[:max_stocks]
        return codes
    except Exception as e:
        print(f"[Warning] Could not fetch index: {e}")
        print("[Info] Falling back to default stock list.")
        return DEFAULT_STOCKS[:max_stocks] if max_stocks else DEFAULT_STOCKS


def fetch_stock_daily(code):
    """
    Fetch daily price/volume data for a single stock.
    Returns DataFrame with columns: [date, open, high, low, close, volume, amount]
    """
    try:
        end = datetime.now()
        start = end - timedelta(days=500)
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start.strftime('%Y%m%d'),
            end_date=end.strftime('%Y%m%d'),
            adjust="qfq"
        )
        if df.empty:
            return None
        
        df.columns = [str(c).strip() for c in df.columns]
        df['date'] = pd.to_datetime(df['日期'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Select core columns
        result = pd.DataFrame()
        result['date'] = df['date']
        result['open'] = pd.to_numeric(df['开盘'], errors='coerce')
        result['high'] = pd.to_numeric(df['最高'], errors='coerce')
        result['low'] = pd.to_numeric(df['最低'], errors='coerce')
        result['close'] = pd.to_numeric(df['收盘'], errors='coerce')
        result['volume'] = pd.to_numeric(df['成交量'], errors='coerce')
        result['amount'] = pd.to_numeric(df['成交额'], errors='coerce')
        result['turnover'] = pd.to_numeric(df['换手率'], errors='coerce')
        
        # Market cap (流通市值 if available)
        if '流通市值' in df.columns:
            result['market_cap'] = pd.to_numeric(df['流通市值'], errors='coerce')
        elif '总市值' in df.columns:
            result['market_cap'] = pd.to_numeric(df['总市值'], errors='coerce')
        else:
            result['market_cap'] = result['close'] * result['volume'] / (result['turnover'] + 0.001)
        
        result.dropna(subset=['close', 'volume'], inplace=True)
        return result
    except Exception:
        return None


def build_panel(stock_codes, max_workers=10, verbose=True):
    """
    Build a cross-sectional panel of daily stock data.
    
    Parameters
    ----------
    stock_codes : list of str
    max_workers : int
    verbose : bool
    
    Returns
    -------
    pd.DataFrame with columns: [date, code, open, high, low, close, volume, amount, turnover, market_cap]
    """
    results = []
    total = len(stock_codes)
    done = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_stock_daily, c): c for c in stock_codes}
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r is not None:
                code = futures[f]
                r['code'] = code
                results.append(r)
            if verbose and done % max(1, total // 5) == 0:
                print(f"  [{done}/{total}] stocks fetched...")
    
    if not results:
        raise RuntimeError("No data could be fetched. Check internet connectivity.")
    
    panel = pd.concat(results, ignore_index=True)
    panel.sort_values(['date', 'code'], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    
    return panel


def generate_synthetic_aiheat(n_days=300):
    """
    Generate synthetic AIHeat time series when real data is unavailable.
    This is a fallback for demonstration purposes.
    
    The synthetic series simulates:
    - Overall upward trend (AI tool adoption)
    - Periods of high volatility (new tool releases, market events)
    - Mean-reverting noise
    
    Returns
    -------
    pd.Series with DatetimeIndex
    """
    np.random.seed(42)
    end = datetime.now()
    dates = pd.date_range(end=end - timedelta(days=n_days), periods=n_days, freq='D')
    
    # Trend + seasonality + noise
    t = np.arange(n_days)
    trend = 0.003 * t  # gradual increase in AI tool interest
    season = 0.3 * np.sin(2 * np.pi * t / 60)  # bi-monthly cycles
    noise = np.random.randn(n_days) * 0.2
    spikes = np.random.choice([0.0, 1.0], size=n_days, p=[0.97, 0.03]) * np.random.uniform(0.5, 1.5, size=n_days)
    
    values = trend + season + noise + spikes
    values = (values - values.mean()) / values.std()  # standardize
    
    return pd.Series(values, index=dates, name='AIHeat')
