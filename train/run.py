"""Run one matched reference (lambda=0) or regularized (lambda=.1) arm.

Use a fresh process for each arm. KWT requires prepare_kwt_cache.py first.
Use a separate run ID for each training pair.
"""
import argparse
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parents[1] / 'experiments/train'
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=('bc8', 'tc14', 'kwt1', 'kwmlp'))
    parser.add_argument('--lambda-max', dest='lambda_max', required=True, type=float, choices=(0.0, 0.1))
    parser.add_argument('--run-id', dest='run_id', default='matched_moment_pairs_v1')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    wrapper = 'torch_pair.py' if args.model in ('bc8',
        'kwmlp') else args.model.replace('kwt1',
        'kwt') + '_pair.py'
    command = [sys.executable,
        str(HERE / wrapper),
        '--lambda-max',
        str(args.lambda_max),
        '--run-id',
        args.run_id,
        '--workers',
        str(args.workers)]
    if args.model in ('bc8', 'kwmlp'):
        command += ['--model', args.model]
    raise SystemExit(subprocess.call(command, cwd=ROOT))
if __name__ == '__main__':
    main()
