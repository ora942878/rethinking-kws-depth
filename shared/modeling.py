"""Explicit checkpoint loading and common block access, without result imports."""
from pathlib import Path
import torch
from .models.bcresnet import BCResNets
from .models.tc_resnet14 import TCResNet14
from .models.kwmlp import KW_MLP
from .models.kwt1_native import NativeKWT
MODELS = ('bc8', 'tc14', 'kwt1', 'kwmlp')

def create_model(kind):
    constructors = {'bc8': lambda: BCResNets(64), 'tc14': TCResNet14, 'kwt1': NativeKWT, 'kwmlp': KW_MLP}
    return constructors[kind]()

def checkpoint_state(payload):
    return payload['model_state_dict'] if 'model_state_dict' in payload else payload

def load_model(kind, checkpoint, device='cpu'):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = create_model(kind)
    model.load_state_dict(checkpoint_state(payload), strict=True)
    return model.float().to(device).eval().requires_grad_(False)

def logits(model, x, kind=None):
    output = model(x)
    return output[0] if isinstance(output, (tuple, list)) else output

def slots(model, kind):
    if kind == 'bc8':
        return [(stage,
            i,
            f'BCBlocks.{s}.{i}') for s,
            stage in enumerate(model.BCBlocks) for i in range(len(stage))]
    blocks = model.layers if kind == 'kwmlp' else model.blocks
    name = 'layers' if kind == 'kwmlp' else 'blocks'
    return [(blocks, i, f'{name}.{i}') for i in range(len(blocks))]

def block(model, kind, number):
    parent, index, _ = slots(model, kind)[number - 1]
    return parent[index]

def put(model, kind, number, module):
    parent, index, _ = slots(model, kind)[number - 1]
    parent[index] = module
