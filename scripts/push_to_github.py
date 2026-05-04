#!/usr/bin/env python3
"""
Create GitHub repo and push local project.
Usage: python scripts/push_to_github.py
Requires GITHUB_TOKEN environment variable.
"""

import os, subprocess, json, requests, sys

token = os.environ.get('GITHUB_TOKEN')
if not token:
    print("Error: Set GITHUB_TOKEN environment variable")
    sys.exit(1)

headers = {
    'Authorization': f'token {token}',
    'Accept': 'application/vnd.github.v3+json'
}

# Get user info
r = requests.get('https://api.github.com/user', headers=headers)
user = r.json()
username = user['login']
print(f'GitHub user: {username}')

# Create repo
repo_name = 'AQC-Reverse-Crowding-Harvest'
r = requests.get(f'https://api.github.com/repos/{username}/{repo_name}', headers=headers)

if r.status_code == 200:
    repo_url = r.json()['html_url']
    print(f'Repo already exists: {repo_url}')
else:
    data = {
        'name': repo_name,
        'description': 'AQC-Reverse Crowding Harvest Factor - AI quant tool heat, crowding unwind, and reverse harvesting for Chinese A-share market',
        'private': False,
        'auto_init': False,
    }
    r = requests.post('https://api.github.com/user/repos', headers=headers, json=data)
    if r.status_code == 201:
        repo_url = r.json()['html_url']
        print(f'Repo created: {repo_url}')
    else:
        print(f'Failed ({r.status_code}): {r.text}')
        sys.exit(1)

# Push local repo
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_dir)

# Add remote
subprocess.run(['git', 'remote', 'add', 'origin', f'https://{username}:{token}@github.com/{username}/{repo_name}.git'],
               capture_output=True)

# Push
result = subprocess.run(['git', 'push', '-u', 'origin', 'master'], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(f'Push stderr: {result.stderr}')
    # Try main branch
    result = subprocess.run(['git', 'branch', '-M', 'main'], capture_output=True)
    result = subprocess.run(['git', 'push', '-u', 'origin', 'main'], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f'Push failed: {result.stderr}')
    else:
        print('Pushed to main!')
else:
    print('Pushed to GitHub!')

print(f'\nYour repo is live at: {repo_url}')
