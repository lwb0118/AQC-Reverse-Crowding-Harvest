#!/usr/bin/env python3
"""
Push project to GitHub via API (works when git protocol is blocked).
"""

import requests, json, base64, os, sys

token = os.environ.get('GITHUB_TOKEN')
if not token:
    print("Error: Set GITHUB_TOKEN")
    sys.exit(1)

owner = 'lwb0118'
repo_name = 'AQC-Reverse-Crowding-Harvest'
headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_url = f'https://api.github.com/repos/{owner}/{repo_name}'

# Get all files
files = []
for root, dirs, filenames in os.walk(project_dir):
    skip_dirs = {'.git', '__pycache__', '.idea', '.vscode'}
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    
    for fname in filenames:
        full = os.path.join(root, fname)
        rel = os.path.relpath(full, project_dir).replace('\\', '/')
        if rel.startswith('.'):
            continue
        with open(full, 'rb') as fh:
            content_b64 = base64.b64encode(fh.read()).decode()
        files.append({'path': rel, 'content': content_b64})

print(f'Total files to push: {len(files)}')

# Push via Contents API (one file at a time, simpler)
pushed = 0
for f in files:
    # Check if file exists
    r = requests.get(f'{base_url}/contents/{f["path"]}', headers=headers)
    
    data = {
        'message': f'Add {f["path"]}',
        'content': f['content'],
        'branch': 'main',
    }
    
    if r.status_code == 200:
        data['sha'] = r.json()['sha']
        data['message'] = f'Update {f["path"]}'
    
    r = requests.put(f'{base_url}/contents/{f["path"]}', headers=headers, json=data)
    
    if r.status_code in [200, 201]:
        pushed += 1
        if pushed % 5 == 0:
            print(f'  [{pushed}/{len(files)}] pushed...')
    else:
        print(f'  Error: {f["path"]} ({r.status_code})')

print(f'\nDone! Pushed {pushed}/{len(files)} files.')
print(f'Repo: https://github.com/{owner}/{repo_name}')
