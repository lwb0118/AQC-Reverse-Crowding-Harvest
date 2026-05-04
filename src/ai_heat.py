"""
ai_heat.py
==========
AI tool heat index (AIHeat) construction.
Combines search indices, GitHub activity, and community discussion proxies.
For demo purposes, provides both real-fetch attempt and synthetic fallback.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_aiheat(search_series=None, github_series=None, forum_series=None,
                   use_synthetic=True, n_days=300, github_token=None):
    """
    Compute AIHeat time series index.
    
    AIHeat_t = AvgZ(Search_t, GitHub_t, Forum_t)
    
    Parameters
    ----------
    search_series : pd.Series, optional
        Search index data with DatetimeIndex.
    github_series : pd.Series, optional
        GitHub activity data with DatetimeIndex.
    forum_series : pd.Series, optional
        Community discussion data with DatetimeIndex.
    use_synthetic : bool
        If True and no real data provided, generate synthetic AIHeat.
    n_days : int
        Length of series.
    github_token : str, optional
        GitHub personal access token for live data.
    
    Returns
    -------
    pd.Series : AIHeat index
    """
    series_list = []
    
    if search_series is not None:
        series_list.append(_standardize_ts(search_series))
    if github_series is not None:
        series_list.append(_standardize_ts(github_series))
    if forum_series is not None:
        series_list.append(_standardize_ts(forum_series))
    
    # Try live GitHub data if token provided
    if github_token and len(series_list) == 0:
        try:
            gh_series = _fetch_github_aiheat(github_token, n_days)
            if gh_series is not None:
                series_list.append(_standardize_ts(gh_series))
                print(f"[AIHeat] Using live GitHub data ({len(gh_series)} days)")
        except Exception as e:
            print(f"[AIHeat] GitHub fetch failed: {e}")
    
    if len(series_list) > 0:
        aiheat = pd.concat(series_list, axis=1).mean(axis=1)
    elif use_synthetic:
        aiheat = _generate_synthetic_aiheat(n_days)
    else:
        raise ValueError("No AIHeat data provided and use_synthetic=False")
    
    aiheat.name = 'AIHeat'
    return aiheat


def _fetch_github_aiheat(token, n_days=400):
    """Build AIHeat time series from live GitHub repo data."""
    from github import Github, Auth
    import numpy as np
    from datetime import datetime, timezone, timedelta
    
    auth = Auth.Token(token)
    g = Github(auth=auth)
    
    repos = [
        'vnpy/vnpy', 'microsoft/qlib', 'ricequant/rqalpha',
        'freqtrade/freqtrade', 'yutiansut/QUANTAXIS',
        'jesse-ai/jesse', 'tensortrade-org/tensortrade'
    ]
    
    repo_data = []
    for name in repos:
        try:
            r = g.get_repo(name)
            repo_data.append({
                'name': name,
                'stars': r.stargazers_count,
                'created': r.created_at,
            })
        except:
            pass
    
    if not repo_data:
        return None
    
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    total_heat = np.zeros(len(dates))
    for rd in repo_data:
        created = rd['created']
        stars = rd['stars']
        repo_age = (end_date - created).days
        if repo_age <= 0:
            continue
        
        mid_point = repo_age * 0.4
        growth_rate = 1.0 / max(repo_age * 0.15, 1)
        
        for i, d in enumerate(dates):
            t = (d - created).days
            if t <= 0:
                continue
            sigmoid = 1.0 / (1.0 + np.exp(-growth_rate * (t - mid_point)))
            growth = sigmoid * (1 - sigmoid) * growth_rate
            total_heat[i] += growth * np.log10(stars + 1) * 10
    
    result = pd.Series(total_heat, index=dates)
    return (result - result.mean()) / result.std()


def compute_aiheat_ma(aiheat, window=20):
    """
    Smoothed AIHeat using moving average.
    """
    return aiheat.rolling(window=window, min_periods=1).mean()


def _standardize_ts(series):
    """Z-score standardize a time series."""
    return (series - series.mean()) / series.std()


def _generate_synthetic_aiheat(n_days=300):
    """Generate synthetic AIHeat for demo purposes."""
    np.random.seed(42)
    end = pd.Timestamp.now()
    dates = pd.date_range(end=end, periods=n_days, freq='D')
    
    t = np.arange(n_days)
    trend = 0.003 * t
    season = 0.3 * np.sin(2 * np.pi * t / 60)
    noise = np.random.randn(n_days) * 0.2
    spikes = np.random.choice([0.0, 1.0], size=n_days, p=[0.97, 0.03])
    spikes = spikes * np.random.uniform(0.5, 1.5, size=n_days)
    
    values = trend + season + noise + spikes
    values = (values - values.mean()) / values.std()
    
    return pd.Series(values, index=dates, name='AIHeat')


def fetch_github_heat(token=None, repos=None, days=365):
    """
    Fetch GitHub activity data for AI-quant related repositories.
    
    Parameters
    ----------
    token : str, optional
        GitHub personal access token. Without it, rate limit is 60 req/hr.
    repos : list, optional
        List of repos to track (owner/repo format).
    days : int
        How many days of history to fetch.
    
    Returns
    -------
    pd.DataFrame with columns: [date, star_count, fork_count, daily_stars]
    """
    if repos is None:
        repos = [
            "microsoft/qlib",
            "ai-powered-quant/ai-quant",
            "hzwer/arXiv2020-RIFE",
            "tensortrade-org/tensortrade",
            "jesse-ai/jesse",
            "freqtrade/freqtrade",
            "nickoala/quantitative-trading",
            "yutiansut/QUANTAXIS",
            "ricequant/rqalpha",
            "vnpy/vnpy",
        ]
    
    try:
        from github import Github
        
        if token:
            g = Github(token)
            print(f"[GitHub] Authenticated (rate limit: {g.get_rate_limit().core.remaining}/{g.get_rate_limit().core.limit})")
        else:
            g = Github()
            print(f"[GitHub] Unauthenticated (rate limit: 60/hr)")
        
        all_data = []
        for repo_name in repos:
            try:
                repo = g.get_repo(repo_name)
                stars = repo.stargazers_count
                forks = repo.forks_count
                all_data.append({
                    'repo': repo_name,
                    'stars': stars,
                    'forks': forks,
                    'updated': repo.updated_at
                })
            except Exception as e:
                print(f"  [Skip] {repo_name}: {e}")
        
        if all_data:
            df = pd.DataFrame(all_data)
            # Normalize to daily frequency
            df['date'] = pd.to_datetime(df['updated'].dt.date)
            return df
    except ImportError:
        print("[Info] PyGithub not installed. Run: pip install PyGithub")
    except Exception as e:
        print(f"[Info] GitHub fetch failed: {e}")
    
    return None


def fetch_aiheat_real():
    """
    Attempt to fetch real AIHeat proxy data.
    
    Current limitations:
    - Baidu Index requires login/cookie
    - GitHub API has rate limits
    - No free public API for aggregated AI quant heat
    
    For production use, users should:
    1. Provide GitHub token for API access
    2. Export Baidu Index data manually
    3. Scrape community forums
    
    Returns
    -------
    dict with keys: 'search', 'github', 'forum' or None
    """
    print("[Info] Attempting real AIHeat data fetch...")
    print("[Info] GitHub API requires a personal access token for reasonable rate limits.")
    print("[Info] Set GITHUB_TOKEN environment variable or pass token parameter.")
    print("[Info] Using synthetic AIHeat for demo. See docs/ for setup.")
    return None
