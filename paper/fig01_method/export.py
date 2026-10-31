"""Export the method SVG with rsvg-convert; no TeX compiler is required."""
import argparse,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('outputs/paper/fig/main_method_v3.pdf'))
p.add_argument('--converter',default='rsvg-convert')
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
subprocess.run([a.converter,
    '--format=pdf',
    '--output',
    str(a.output),
    str(Path(__file__).with_name('source.svg'))],
    check=True)
