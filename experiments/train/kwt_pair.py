'KWT-1 source training with cached author MFCC views and moment loss.'
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import types
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
AUTHOR = HERE / '_author'
DEFAULT_RUN = 'matched_moment_pairs_v1'
CACHE = Path(os.environ.get('KWS_KWT_CACHE', str(ROOT / 'outputs/cache/kwt1_author'))).expanduser().resolve()
SEED = 20261007
TOTAL = 23438

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lambda-max', type=float, choices=[0.0, 0.1], required=True)
    p.add_argument('--run-id', default=DEFAULT_RUN)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if os.environ.get('PYTHONHASHSEED') != str(SEED):
        os.environ['PYTHONHASHSEED'] = str(SEED)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    tag = 'lambda_0' if args.lambda_max == 0 else 'lambda_0p1'
    stage = 'full'
    out = ROOT / 'outputs/train' / args.run_id / 'kwt1' / tag / stage
    weights = ROOT / 'weights' / args.run_id / 'kwt1' / tag / stage
    out.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    started = time.time()
    state = dict(status='INITIALIZING',
        pid=os.getpid(),
        lambda_max=args.lambda_max,
        seed=SEED,
        step=0,
        total=TOTAL,
        started=started,
        cache=str(CACHE),
        batch=512,
        source='ARM native TensorFlow author code',
        change='Moment loss weight only; shared cached frontend from step zero')
    atomic(out / 'status.json', state)
    processors = []
    try:
        os.environ.update(TF_USE_LEGACY_KERAS='1',
            TF_CPP_MIN_LOG_LEVEL='2',
            TF_NUM_INTRAOP_THREADS=str(max(1,
            args.workers)),
            TF_NUM_INTEROP_THREADS='2',
            OMP_NUM_THREADS='1',
            TF_FORCE_GPU_ALLOW_GROWTH='true')
        sys.path.insert(0, str(AUTHOR))
        import kwt_cached
        from experiment_common import source_info
        source = source_info('kwt1')
        import numpy as np
        random.seed(SEED)
        np.random.seed(SEED)
        tf = kwt_cached.tf_setup()
        from kws_streaming.train import train as author_train
        from kws_streaming.models import models, kws_transformer
        import kwt_adapter
        flags = types.SimpleNamespace(**kwt_cached.config())
        flags.start_checkpoint = ''
        flags.train_dir = str(weights) + '/'
        flags.summaries_dir = str(out / 'tensorboard')
        flags.how_many_training_steps = str(state['total'])
        flags.preprocess = 'mfcc'
        flags.verbosity = 'WARNING'
        data = kwt_cached.rows()
        state['source'] = source

        class Processor:

            def __init__(self, flags):
                self.words_list = ['_silence_', '_unknown_'] + data['classes'][2:]
                self.x = {s: [np.load(CACHE / f'{s}_v{v}.npy') for v in range(4 if s == 'train' else 1)] for s in data['rows']}
                self.y = {s: np.load(CACHE / f'{s}_labels.npy') for s in data['rows']}
                self.rng = np.random.default_rng(SEED)
                self.pool = ThreadPoolExecutor(max_workers=1)
                self.next = None
                self.next_offset = None
                processors.append(self)

            def set_size(self, mode):
                return len(self.y[{'training': 'train', 'validation': 'validation', 'testing': 'test'}[mode]])

            def build_batch(self, n, offset):
                ids = np.arange(offset, offset + n) % len(self.y['train'])
                views = self.rng.integers(0, 4, len(ids))
                x = np.empty((n, 98, 40), np.float32)
                for v in range(4):
                    m = views == v
                    x[m] = self.x['train'][v][ids[m]]
                return (x, self.y['train'][ids])

            def get_data(self, n, offset, flags, bgf, bgv, shift, mode, resample, volume, sess):
                s = {'training': 'train', 'validation': 'validation', 'testing': 'test'}[mode]
                if s != 'train':
                    return (self.x[s][0][offset:offset + n], self.y[s][offset:offset + n])
                if self.next is None:
                    result = self.build_batch(n, offset)
                else:
                    result = self.next.result()
                self.next_offset = offset + n
                self.next = self.pool.submit(self.build_batch, n, self.next_offset)
                return result
        author_train.input_data.AudioProcessor = Processor
        model_source = Path(kws_transformer.__file__).read_text()
        needle = '  net = input_audio\n'
        model_source = model_source.replace(needle,
            '  from kws_streaming.layers.spectrogram_augment import SpecAugment\n  net = SpecAugment(time_masks_number=flags.time_masks_number, time_mask_max_size=flags.time_mask_max_size, frequency_masks_number=flags.frequency_masks_number, frequency_mask_max_size=flags.frequency_mask_max_size)(input_audio)\n')
        module = types.ModuleType('kws_streaming.models.kws_transformer_ablation')
        module.__file__ = str(out / 'author_model_cached.py')
        Path(module.__file__).write_text(model_source)
        exec(compile(model_source, module.__file__, 'exec'), module.__dict__)
        models.MODELS['kws_transformer'] = module.model
        step_started = [None]
        last_weight = [None]
        train_started = [None]
        durations = deque(maxlen=200)
        state['best_validation_accuracy'] = 0.0

        def record(event, **kw):
            state.update(kw)
            state['updated'] = time.time()
            state['elapsed_seconds'] = time.time() - started
            if train_started[0] and state['step'] > 0:
                state['training_elapsed_seconds'] = time.time() - train_started[0]
                state['eta_seconds'] = state['training_elapsed_seconds'] / state['step'] * (state['total'] - state['step'])
            if durations:
                state['mean_optimizer_step_seconds'] = float(np.mean(durations))
            atomic(out / 'status.json', state)
            with (out / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(dict(event=event, step=state['step'], time=time.time(), **kw)) + '\n')
            print(json.dumps(dict(event=event,
                step=state['step'],
                total=state['total'],
                eta_seconds=state.get('eta_seconds'),
                **kw)),
                flush=True)

        def initialized(model, sess):
            train_started[0] = time.time()

        def before_step(step, sess, model, inputs):
            state['step'] = step
            w = args.lambda_max * min(1.0, max(0.0, (step / (36923 / 512) - 5) / 15))
            ws = [v for v in tf.global_variables() if v.name.endswith('moment_regularization/weight:0')]
            if w != last_weight[0]:
                tf.keras.backend.set_value(ws[0], w)
                last_weight[0] = w
            state['lambda_value'] = w
            step_started[0] = time.perf_counter()

        def after_step(step, result):
            durations.append(time.perf_counter() - step_started[0])
            if step % 50 == 0 or step == state['total']:
                record('STEP',
                    status='TRAINING',
                    loss=float(result[0]),
                    training_accuracy=100 * float(result[-1]),
                    lambda_value=last_weight[0])

        def after_validation(step, accuracy, best):
            record('VALIDATION',
                validation_accuracy=100 * float(accuracy),
                best_validation_accuracy=100 * float(best))

        def final_test(model, processor, flags, sess):
            model.load_weights(flags.train_dir + 'best_weights').expect_partial()
            cm = np.zeros((12, 12), dtype=np.int64)
            for i in range(0, processor.set_size('testing'), flags.batch_size):
                x, y = processor.get_data(flags.batch_size, i, flags, 0, 0, 0, 'testing', 0, 0, sess)
                pred = model.predict_on_batch(x).argmax(axis=1)
                np.add.at(cm, (y, pred), 1)
            atomic(out / 'best_test.json',
                dict(samples=int(cm.sum()),
                accuracy_percent=100 * float(np.trace(cm)) / cm.sum(),
                confusion_matrix=cm.tolist()))
            record('BEST_TEST',
                best_test_accuracy=100 * float(np.trace(cm)) / cm.sum(),
                test_samples=int(cm.sum()))
        kwt_adapter.before_step = before_step
        kwt_adapter.after_step = after_step
        kwt_adapter.after_validation = after_validation
        kwt_adapter.final_test = final_test
        kwt_adapter.ablation_initialized = initialized
        source = Path(author_train.__file__).read_text()
        needle = '  sess.run(tf.global_variables_initializer())'
        source = source.replace(needle,
            '  model._make_train_function()\n  model._make_test_function()\n  model._make_predict_function()\n' + needle + '\n  from kwt_adapter import ablation_initialized\n  ablation_initialized(model,sess)')
        path = out / 'author_train_ablation.py'
        path.write_text(source)
        spec = importlib.util.spec_from_file_location('author_train_ablation', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        atomic(out / 'config.json', dict(flags=vars(flags), lambda_max=args.lambda_max, seed=SEED))
        mod.train(flags)
        record('COMPLETE', status='COMPLETE')
        atomic(out / 'COMPLETE.json', state)
    except Exception as e:
        state.update(status='FAILED', error=repr(e), traceback=traceback.format_exc(), updated=time.time())
        atomic(out / 'status.json', state)
        raise
    finally:
        for proc in processors:
            proc.pool.shutdown(wait=True)
if __name__ == '__main__':
    main()
