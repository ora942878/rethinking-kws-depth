"BC8 and KW-MLP source training with the authors' loops and moment loss."
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
HERE = Path(__file__).resolve().parent
AUTHOR = HERE / '_author'
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
SEED = 20261007


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)

def atomic_checkpoint(path, value):
    """Keep the preceding completed checkpoint intact until its replacement is durable."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def seed_numba(seed):
    """Seed Numba's independent NumPy RNG inside compiled code, not Python."""
    import numpy as np
    from numba import njit

    @njit(cache=False)
    def compiled_seed(value):
        np.random.seed(value)
    compiled_seed(int(seed))

def seed_worker(worker_id):
    import numpy as np
    import torch
    seed = torch.initial_seed() % 2 ** 32
    if os.environ.get('MATCHED_MOMENT_PAIR_MODEL') == 'kwmlp':
        seed_numba(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_loader(dataset, batch_size, split, monitor):
    split_id = {'train': 0, 'validation': 1, 'test': 2}[split]
    worker_generator = torch.Generator().manual_seed(SEED + 100 + split_id)
    sampler = torch.utils.data.RandomSampler(dataset,
        generator=torch.Generator().manual_seed(SEED + 200)) if split == 'train' else None
    return torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=monitor.args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=worker_generator)

class Monitor:

    def __init__(self, args, common, recipe):
        self.args = args
        self.kind = args.model
        self.variant = 'lambda_0' if args.lambda_max == 0 else 'lambda_0p1'
        self.manifest = common.manifest(args.model)
        self.rows = self.manifest['rows']
        self.spe = math.ceil(len(self.rows['train']) / recipe['batch_size'])
        self.total = self.spe * recipe['epochs']
        self.step = 0
        self.best = -1.0
        self.started = time.time()
        self.values = {}
        relative = Path(args.run_id) / args.model / self.variant / 'full'
        self.out = ROOT / 'outputs/train' / relative
        self.weights = ROOT / 'weights/retrained' / relative
        self.out.mkdir(parents=True, exist_ok=True)
        self.weights.mkdir(parents=True, exist_ok=True)
        self.config = dict(model=args.model, seed=SEED, lambda_max=args.lambda_max, recipe=recipe)
        atomic_json(self.out / 'config.json', self.config)

    def update(self, status='RUNNING', **values):
        self.values.update(values)
        atomic_json(self.out / 'status.json',
            dict(self.values,
            status=status,
            step=self.step,
            total_steps=self.total,
            epoch=self.step / self.spe,
            elapsed_seconds=time.time() - self.started))

    def record(self, event, **values):
        with (self.out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(dict(event=event, step=self.step, **values)) + '\n')
        self.update(**values)

    def initialize(self, model):
        self.model = model

    def objective(self, model, collector, ce):
        reg, active_pairs = collector.loss()
        weight = self.args.lambda_max * min(1.0, max(0.0, ((self.step + 1) / self.spe - 5) / 15))
        total = ce + weight * reg
        self.last_loss = dict(ce=float(ce.detach()),
            moment_loss=float(reg.detach()),
            total_loss=float(total.detach()),
            lambda_value=weight)
        return total

    def checkpoint_metadata(self):
        return dict(training_config=self.config)

    def finish(self, **values):
        self.update('COMPLETE', **values)

def run_bc8(args, common, old):
    recipe = dict(epochs=200,
        batch_size=100,
        optimizer='SGD',
        momentum=0.9,
        weight_decay=0.001,
        learning_rate='5 epoch warmup to .1, cosine to 0')
    mon = Monitor(args, common, recipe)
    sys.path.insert(0, str(ROOT / 'third_party/bc8'))
    upstream = old.load_entry(ROOT / 'third_party/bc8/main.py', 'matched_author_bc8_main')
    from utils import Padding, Preprocess
    from bcresnet import BCResNets

    class Trainer(upstream.Trainer):

        def __init__(self):
            self.device = torch.device('cuda')
            self.ver, self.tau, self.epoch, self.best = (2, 8, 0, -1.0)
            for name, split in (('train', 'train'), ('valid', 'validation'), ('test', 'test')):
                dataset = old.AudioRows(mon.rows[split], Padding())
                setattr(self, name + '_dataset', dataset)
                setattr(self, name + '_loader', make_loader(dataset, 100, split, mon))
            self.preprocess_train = Preprocess(None, self.device, specaug=True, frequency_masking_para=7)
            self.preprocess_test = Preprocess(None, self.device)
            noise = []
            for path in sorted((common.dataset_root() / '_background_noise_').glob('*.wav')):
                audio, sr = old.sf.read(path, dtype='float32')
                noise.append(torch.from_numpy(audio).unsqueeze(0))
            self.preprocess_train.background_noise = noise
            self.preprocess_test.background_noise = noise
            self.model = BCResNets(64).to(self.device)
            self.collector = common.TorchMoments(self.model, 'bc8')
            mon.initialize(self.model)

        def Test(self, dataset, loader, augment):
            accuracy = super().Test(dataset, loader, augment)
            if dataset is self.valid_dataset:
                self.epoch += 1
                improved = accuracy > self.best
                if improved:
                    self.best = float(accuracy)
                payload = dict(model_state_dict=self.model.state_dict(),
                    epoch=self.epoch,
                    validation_accuracy=float(accuracy),
                    best_validation_accuracy=self.best,
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict={'type': 'author_iteration_warmup_cosine',
                    'iterations': mon.step,
                    'total_iterations': mon.total,
                    'warmup_iterations': 5 * mon.spe,
                    'initial_lr': 0.1,
                    'minimum_lr': 0.0},
                    completed_epoch_boundary=mon.step == self.epoch * mon.spe,
                    **mon.checkpoint_metadata())
                atomic_checkpoint(mon.weights / 'last.pth', payload)
                if improved:
                    atomic_checkpoint(mon.weights / 'best.pth', payload)
                mon.record('EPOCH',
                    validation_accuracy=float(accuracy),
                    best_validation_accuracy=self.best,
                    completed_epoch=self.epoch)
            else:
                mon.record('TEST', test_accuracy=float(accuracy))
            return accuracy
    trainer = Trainer()
    original_torch = upstream.torch
    original_sgd = original_torch.optim.SGD

    def captured_sgd(*positional, **keywords):
        trainer.optimizer = original_sgd(*positional, **keywords)
        return trainer.optimizer

    class CapturedOptim:
        SGD = staticmethod(captured_sgd)

        def __getattr__(self, name):
            return getattr(original_torch.optim, name)

    class CapturedTorch:
        optim = CapturedOptim()

        def __getattr__(self, name):
            return getattr(original_torch, name)
    upstream.torch = CapturedTorch()
    original_ce = upstream.F.cross_entropy

    def objective(logits, labels, *positional, **keywords):
        ce = original_ce(logits, labels, *positional, **keywords)
        if trainer.model.training and torch.is_grad_enabled():
            value = mon.objective(trainer.model, trainer.collector, ce)
            mon.step += 1
            if mon.step % 25 == 0:
                mon.record('STEP', **mon.last_loss)
            return value
        return ce
    upstream.F.cross_entropy = objective
    try:
        trainer()
        last_accuracy = float(mon.values['test_accuracy'])
        trainer.model.load_state_dict(torch.load(mon.weights / 'best.pth',
            map_location='cpu',
            weights_only=False)['model_state_dict'])
        trainer.model.eval()
        with torch.no_grad():
            accuracy = float(trainer.Test(trainer.test_dataset, trainer.test_loader, False))
        mon.finish(best_test_accuracy=accuracy,
            last_test_accuracy=last_accuracy,
            author_loop='bc8/main.py:Trainer.__call__')
    except Exception as exc:
        mon.update('FAILED', error=repr(exc))
        raise
    finally:
        upstream.F.cross_entropy = original_ce
        upstream.torch = original_torch

def run_kwmlp(args, common, old):
    import yaml
    cfg = yaml.safe_load((ROOT / 'third_party/kwmlp/sample_configs/base_config.yaml').read_text())
    cfg['hparams'].update(seed=SEED, device='cuda', start_epoch=0, restore_ckpt=None, n_epochs=140)
    recipe = dict(epochs=140, batch_size=256, hparams=cfg['hparams'], cache=2)
    recipe = json.loads(json.dumps(recipe))
    mon = Monitor(args, common, recipe)
    sys.path.insert(0, str(ROOT / 'third_party/kwmlp'))
    upstream = old.load_entry(ROOT / 'third_party/kwmlp/train.py', 'matched_author_kwmlp_train')
    import utils.trainer as trainer
    import utils.misc as misc
    from utils.dataset import GoogleSpeechDataset
    cfg['data_root'] = str(common.dataset_root())
    cfg['hparams'].update(seed=SEED, device='cuda', start_epoch=0, restore_ckpt=None, n_epochs=140)
    cfg['exp'].update(exp_dir=str(mon.weights.parent),
        exp_name=mon.weights.name,
        wandb=False,
        n_workers=args.workers,
        cache=2,
        log_to_file=True)
    for split, key in (('train',
        'train_list_file'),
        ('validation',
        'val_list_file'),
        ('test',
        'test_list_file')):
        path = mon.out / (split + '_list.txt')
        path.write_text('\n'.join(((common.dataset_root() / rel).as_posix() for rel,
            label in mon.rows[split])) + '\n',
            encoding='utf-8')
        cfg[key] = str(path)
    labels = {str(i): value for i, value in enumerate(mon.manifest['classes'])}
    atomic_json(mon.out / 'label_map.json', labels)
    cfg['label_map'] = str(mon.out / 'label_map.json')
    state = {}
    loader_calls = iter(('train', 'validation', 'test'))

    def get_loader(data_list, config, train=True):
        split = next(loader_calls)
        dataset = GoogleSpeechDataset(data_list,
            config['hparams']['audio'],
            label_map=labels,
            aug_settings=config['hparams']['augment'] if train else None,
            cache=config['exp']['cache'])
        loader = make_loader(dataset, 256, split, mon)
        state[split + '_loader'] = loader
        return loader
    original_model = upstream.get_model

    def get_model(config):
        net = original_model(config)
        state['net'] = net
        state['collector'] = common.TorchMoments(net, 'kwmlp')
        mon.initialize(net)
        return net
    criterion_type = upstream.LabelSmoothingLoss

    class Objective(criterion_type):

        def forward(self, logits, labels):
            ce = super().forward(logits, labels)
            if state['net'].training and torch.is_grad_enabled():
                return mon.objective(state['net'], state['collector'], ce)
            return ce
    original_batch = trainer.train_single_batch

    def train_batch(*positional, **keywords):
        state['optimizer'] = positional[3]
        state['criterion'] = positional[4]
        value = original_batch(*positional, **keywords)
        mon.step += 1
        if mon.step % 25 == 0:
            mon.record('STEP', **mon.last_loss)
        return value
    original_log = trainer.log

    def log(values, step, config):
        original_log(values, step, config)
        if 'val_acc' in values:
            mon.best = max(mon.best, 100 * values['val_acc'])
            mon.record('EPOCH',
                validation_accuracy=100 * values['val_acc'],
                best_validation_accuracy=mon.best,
                completed_epoch=values['epoch'] + 1)

    def pipeline_log(values, step, config):
        misc.log(values, step, config)
        if 'test_acc_best' in values:
            mon.record('BEST_TEST', best_test_accuracy=100 * values['test_acc_best'])
        if 'test_acc_last' in values:
            mon.record('LAST_TEST', last_test_accuracy=100 * values['test_acc_last'])
    original_save = trainer.save_model

    def save_model(epoch, val_acc, save_path, net, optimizer=None, scheduler=None, log_file=None):
        author_temporary = Path(str(save_path) + '.author.tmp')
        original_save(epoch, val_acc, str(author_temporary), net, optimizer, scheduler, log_file)
        payload = torch.load(author_temporary, map_location='cpu', weights_only=False)
        payload.update(mon.checkpoint_metadata())
        atomic_checkpoint(save_path, payload)
        author_temporary.unlink()
    upstream.get_loader, upstream.get_model = (get_loader, get_model)
    upstream.LabelSmoothingLoss, upstream.log = (Objective, pipeline_log)
    trainer.train_single_batch, trainer.log, trainer.save_model = (train_batch, log, save_model)
    try:
        upstream.training_pipeline(cfg)
        mon.finish(author_loop='kwmlp/train.py:training_pipeline and utils/trainer.py:train')
    except Exception as exc:
        mon.update('FAILED', error=repr(exc))
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('bc8', 'kwmlp'), required=True)
    parser.add_argument('--lambda-max', type=float, choices=(0.0, 0.1), required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ['MATCHED_MOMENT_PAIR_MODEL'] = args.model
    if os.environ.get('PYTHONHASHSEED') != str(SEED):
        os.environ['PYTHONHASHSEED'] = str(SEED)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    global torch, np
    import torch
    import numpy as np
    torch.set_num_threads(4)
    if args.model == 'kwmlp':
        seed_numba(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    sys.path.insert(0, str(AUTHOR))
    import experiment_common as common
    import torch_data as old
    (run_bc8 if args.model == 'bc8' else run_kwmlp)(args, common, old)
if __name__ == '__main__':
    main()
