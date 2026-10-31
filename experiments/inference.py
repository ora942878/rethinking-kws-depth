"""Single-sample network FLOPs and MACs, excluding audio preprocessing."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from shared.modeling import load_model,logits
from shared.structure import geometry,build_case
from shared.cost import inference_cost
from experiments.budget import DEFAULT_DROPS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=tuple(DEFAULT_DROPS),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    x=torch.from_numpy(np.array(np.load(a.prepared_dir/'test_x.npy',mmap_mode='r')[:1]))
    original=load_model(a.model,a.checkpoint);g=geometry(original,a.model,x);rows=[]
    for name,drops in [('original',[]),('direct',DEFAULT_DROPS[a.model])]:
        model,_=build_case(original,a.model,dict(family='drop',drops=drops,copies=[],mode='F',scope='none'),g)
        count=inference_cost(model,x,lambda x:logits(model,x,a.model));count.pop('operator_report',None)
        rows.append(dict(model=a.model,structure=name,deleted_blocks=drops,**count))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
