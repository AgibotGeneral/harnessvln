#!/usr/bin/env python3
"""Run offline release checks: Python 3.9 syntax and shell syntax; no GPU or API required."""
import ast
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / 'run_mp.py', ROOT / 'server.py']
    for folder in ('vlnce_baselines', 'habitat_extensions', 'navdp', 'tools'):
        paths.extend((ROOT / folder).rglob('*.py'))
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 9))
    print(f'PASS Python 3.9 syntax: {len(paths)} source files', flush=True)
    for path in sorted((ROOT / 'eval_scripts').rglob('*.sh')):
        subprocess.run(['bash', '-n', str(path)], check=True, cwd=ROOT)
    print('PASS shell syntax', flush=True)
    print('PASS all offline release checks', flush=True)


if __name__ == '__main__':
    main()
