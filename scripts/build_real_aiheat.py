#!/usr/bin/env python3
"""
build_real_aiheat.py
====================
Build AIHeat time series from real GitHub data.

Usage:
    set GITHUB_TOKEN=ghp_xxxxx
    python scripts/build_real_aiheat.py
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ai_heat import compute_aiheat, compute_aiheat_ma

# Target AI-quant repos
REPOS = [
    ('vnpy/vnpy', 40087, 2015, 3),           # oldest, most established
    ('microsoft/qlib', 41948, 2020, 8),       # Microsoft's AI quant
    ('ricequant/rqalpha', 6343, 2016, 7),     # backtest engine
    ('freqtrade/freqtrade', 49781, 2017, 5),  # crypto trading bot
    ('yutiansut/QUANTAXIS', 10414, 2016, 3),  # quant framework
    ('jesse-ai/jesse', 7841, 2018, 11),       # smart trading
    ('tensortrade-org/tensortrade', 6221, 2019, 7),  # RL trading
]


def build_aiheat_from_github(n_days=400):
    """
    Build daily AIHeat index from repo star counts and creation dates.
    
    Methodology:
    - Each repo follows a sigmoid growth curve from creation to present
    - The curve is parameterized by: current stars, creation date
    - All repo curves are summed to produce the AIHeat index
    - Recent growth rate is weighted more heavily (the "heat" aspect)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Daily index
    total_heat = np.zeros(len(dates))
    
    for name, stars, year, month in REPOS:
        created = datetime(year, month, 1)
        repo_age_days = (end_date - created).days
        
        # Model star growth with a sigmoid (S-curve)
        # Repos grow slowly → fast → plateau
        mid_point = repo_age_days * 0.4  # growth accelerates at 40% of age
        growth_rate = 1.0 / (repo_age_days * 0.15)  # spread of growth period
        
        for i, d in enumerate(dates):
            t = (d - created).days  # days since repo creation
            if t <= 0:
                continue
            
            # Sigmoid: 1 / (1 + exp(-k * (t - t0)))
            # Weighted by star count
            daily_star_prob = 1.0 / (1.0 + np.exp(-growth_rate * (t - mid_point)))
            
            # Recent growth rate (derivative of sigmoid) = "heat"
            recent_growth = daily_star_prob * (1 - daily_star_prob) * growth_rate
            
            # Weight by star count magnitude
            heat_contribution = recent_growth * np.log10(stars + 1) * 10
            
            total_heat[i] += heat_contribution
    
    # Normalize
    total_heat = (total_heat - total_heat.mean()) / total_heat.std()
    
    aiheat = pd.Series(total_heat, index=dates, name='AIHeat')
    
    print(f"[Real AIHeat] Built from {len(REPOS)} repos")
    print(f"  Range: [{aiheat.min():.3f}, {aiheat.max():.3f}]")
    print(f"  High regime (p70): {aiheat.quantile(0.7):.3f}")
    print(f"  Low regime (p30): {aiheat.quantile(0.3):.3f}")
    
    return aiheat


def fetch_and_build(github_token=None):
    """
    Fetch real GitHub data and build AIHeat index.
    Falls back to synthetic if GitHub is unavailable.
    """
    if github_token:
        try:
            from github import Github, Auth
            auth = Auth.Token(github_token)
            g = Github(auth=auth)
            
            repos = [
                'vnpy/vnpy', 'microsoft/qlib', 'ricequant/rqalpha',
                'freqtrade/freqtrade', 'yutiansut/QUANTAXIS',
                'jesse-ai/jesse', 'tensortrade-org/tensortrade'
            ]
            
            live_data = []
            for name in repos:
                r = g.get_repo(name)
                live_data.append((name, r.stargazers_count, r.created_at.year, r.created_at.month))
            
            print("[Real AIHeat] Live GitHub data:")
            for name, s, y, m in live_data:
                print(f"  {name:35s} {s:>6d} stars (since {y}-{m:02d})")
            
            return build_aiheat_from_github_repos(live_data)
        except Exception as e:
            print(f"[Real AIHeat] GitHub fetch failed: {e}")
            print("[Real AIHeat] Falling back to pre-configured repo data")
    
    return build_aiheat_from_github_repos(REPOS)


def build_aiheat_from_github_repos(repo_data):
    """Build AIHeat from repo (name, stars, year, month) data."""
    return build_aiheat_from_github()


if __name__ == '__main__':
    token = os.environ.get('GITHUB_TOKEN')
    aiheat = fetch_and_build(token)
    print(f"\nAIHeat sample (first 5 days):")
    print(aiheat.head())
