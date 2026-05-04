# AIHeat Data Acquisition Guide

The AIHeat index requires external data. This guide explains how to set up real data sources.

## Option 1: GitHub API (Recommended)

1. Create a GitHub Personal Access Token:
   - Go to GitHub.com → Settings → Developer settings → Personal access tokens → Fine-grained tokens
   - Generate a token with `public_repo` access
   
2. Set the token as environment variable:
   ```bash
   set GITHUB_TOKEN=ghp_your_token_here
   ```
   Or pass it directly in code.

3. The system will fetch star/fork counts from popular AI-quant repos.

## Option 2: Baidu Index (Baidu 指数)

1. Go to https://index.baidu.com
2. Search for keywords: `AI量化`, `量化交易`, `ChatGPT选股`, `AI Agent投资`
3. Export the trend data as CSV
4. Place CSV files in `data/external/`
5. Use `pd.read_csv()` to load in `ai_heat.py`

## Option 3: Community Data

- 雪球: Search for AI-quant related discussions
- 知乎: Track question/answer counts
- GitHub: Use GraphQL API for aggregated stats

## Data Format

All data sources should produce a time series with DatetimeIndex.
The series will be Z-score standardized and averaged to produce AIHeat.

## 百度指数 (推荐 & 免费)

1. 打开 https://index.baidu.com (需登录百度账号)
2. 搜索以下关键词，每个单独搜:
   量化交易、AI量化、自动选股、量化策略、ChatGPT选股
3. 点击右上角"下载数据" → "按天" → 下载CSV
4. 将CSV文件放入 data/external/baidu_index/
5. 文件名格式: baidu_量化交易.csv, baidu_AI量化.csv

项目内置读取函数:
from src.ai_heat import compute_aiheat_from_baidu
aiheat = compute_aiheat_from_baidu('data/external/baidu_index/')

