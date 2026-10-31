"""Deletion, whole-block reuse and recovery for the four paper models."""
import argparse, json
from pathlib import Path
import torch
from shared.data import load_prepared
from shared.modeling import load_model
from shared.structure import geometry,cases,build_case
from shared.recovery import recover
from shared.metrics import evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['bc8','tc14','kwt1','kwmlp'],required=True)
    p.add_argument('--arm',choices=['lambda_0','lambda_0p1'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--families',
        nargs='+',
        choices=['drop',
        'reuse',
        'norm',
        'affine',
        'reuse_norm'],
        default=['drop',
        'reuse',
        'norm',
        'affine'])
    p.add_argument('--max-blocks',type=int,default=4)
    p.add_argument('--scope',choices=['suffix','all'],default='suffix')
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261007)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=load_prepared(a.prepared_dir)
    original=load_model(a.model,a.checkpoint,a.device)
    g=geometry(original,a.model,data['test'][0])
    planned=cases(a.model,g,a.families,a.max_blocks,a.scope)
    ids=torch.randperm(len(data['train'][1]),generator=torch.Generator().manual_seed(20261007))[:12800]
    train=tuple(x[ids] for x in data['train'])
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'config.json').write_text(json.dumps(vars(a),default=str,indent=2))
    with (a.output/'results.jsonl').open('w') as stream:
        for i,case in enumerate(planned,1):
            model,norms=build_case(original,a.model,case,g)
            recovery=recover(model,a.model,norms,case,train,100,a.device,128,.001,.1)
            validation,_=evaluate(model,a.model,*data['validation'],128)
            test,_=evaluate(model,a.model,*data['test'],128)
            if case['family']=='baseline': baseline=test['accuracy_percent']
            row=dict(case,model=a.model,arm=a.arm,k=len(case['drops'])+len(case['copies']),
                accuracy_percent=test['accuracy_percent'],
                    accuracy_change_pp=test['accuracy_percent']-baseline,
                macro_f1=test['macro_f1'],parameters=sum(x.numel() for x in model.parameters()),
                test=test,validation=validation,recovery=recovery)
            stream.write(json.dumps(row)+'\n');stream.flush()
            print(f"{i}/{len(planned)} {case['family']} {case['mode']} {row['accuracy_percent']:.4f}",
                flush=True)

if __name__=='__main__':main()
