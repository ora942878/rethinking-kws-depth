"""Generate paper figures and result tables from the bundled experiment records."""
import argparse,shutil
from pathlib import Path
from . import build

def generate(number=None,output=None):
    if output:build.PAPER=Path(output).resolve()
    build.load()
    if number==1:
        for ext in ('svg','pdf'):
            src=build.ROOT/'paper/fig01_method'/('source.svg' if ext=='svg' else 'figure.pdf')
            shutil.copy2(src,build.PAPER/'fig'/('main_method_v3.'+ext))
    elif number in (2,3,4,5,6):build.figures(selected={number-1})
    elif number in (7,8):build.paired_figures(number)
    else:
        build.figures();build.tables()
        for ext in ('svg','pdf'):
            src=build.ROOT/'paper/fig01_method'/('source.svg' if ext=='svg' else 'figure.pdf')
            shutil.copy2(src,build.PAPER/'fig'/('main_method_v3.'+ext))
    print('Output:',build.PAPER)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure',type=int,choices=range(1,9))
    p.add_argument('--output',type=Path)
    a=p.parse_args();generate(a.figure,a.output)
