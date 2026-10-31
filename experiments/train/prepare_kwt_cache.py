"""Prepare immutable native-frontend KWT features shared by both loss arms.

Default cache: outputs/cache/kwt1_author. Override with KWS_KWT_CACHE.
This command preprocesses audio; it does not run optimizer updates.
"""
import argparse
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE / '_author'))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    from kwt_cached import cache
    cache(args.workers)
if __name__ == '__main__':
    main()
