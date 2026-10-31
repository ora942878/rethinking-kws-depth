"""Parallel author waveform preprocessing and cached MFCC for paired KWT training.

The cache has one clean and three augmented training views. Fresh online
SpecAugment remains active. This finite-view schedule is recorded separately
from the original online waveform augmentation run.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = Path(os.environ.get('KWS_KWT_CACHE', str(ROOT / 'outputs/cache/kwt1_author'))).expanduser().resolve()
SOURCE = ROOT / 'third_party/kwt1'
from experiment_common import dataset_root, manifest, source_info

def atomic(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(v, indent=2))
    tmp.replace(p)



def config():
    from kws_streaming.train import base_parser
    from kws_streaming.models import model_flags, kws_transformer
    parser = base_parser.base_parser()
    sub = parser.add_subparsers(dest='model_name')
    kws_transformer.model_parameters(sub.add_parser('kws_transformer'))
    args = ['--data_url',
        '',
        '--data_dir',
        str(dataset_root()),
        '--train_dir',
        str(ROOT / 'weights/_cache_config_unused') + '/',
        '--mel_upper_edge_hertz',
        '7600',
        '--optimizer',
        'adamw',
        '--lr_schedule',
        'cosine',
        '--how_many_training_steps',
        '23438',
        '--eval_step_interval',
        '72',
        '--warmup_epochs',
        '10',
        '--l2_weight_decay',
        '0.1',
        '--learning_rate',
        '0.001',
        '--batch_size',
        '512',
        '--label_smoothing',
        '0.1',
        '--window_size_ms',
        '30.0',
        '--window_stride_ms',
        '10.0',
        '--mel_num_bins',
        '80',
        '--dct_num_features',
        '40',
        '--resample',
        '0.15',
        '--train',
        '1',
        '--use_spec_augment',
        '1',
        '--time_masks_number',
        '2',
        '--time_mask_max_size',
        '25',
        '--frequency_masks_number',
        '2',
        '--frequency_mask_max_size',
        '7',
        '--pick_deterministically',
        '1',
        'kws_transformer',
        '--num_layers',
        '12',
        '--heads',
        '1',
        '--d_model',
        '64',
        '--mlp_dim',
        '256',
        '--dropout1',
        '0.',
        '--attention_type',
        'time']
    flags = model_flags.update_flags(parser.parse_args(args))
    return vars(flags)

def rows():
    data = manifest('kwt1')
    rng = random.Random(20261007)
    for split in ['train', 'validation', 'test']:
        rng.shuffle(data['rows'][split])
    return data

def cache_metadata():
    return json.loads((CACHE / 'COMPLETE.json').read_text(encoding='utf-8'))

def tf_setup(cpu=False):
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if cpu:
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    else:
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
        os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    os.environ['OMP_NUM_THREADS'] = '1'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sys.path.insert(0, str(SOURCE))
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    return tf

def worker_init():
    global WF, WP, WS, WN
    tf = tf_setup(True)
    import numpy as np
    from kws_streaming.data import input_data
    WF = SimpleNamespace(**config())
    data = rows()

    def index(self, *a, **kw):
        words = [input_data.SILENCE_LABEL, input_data.UNKNOWN_WORD_LABEL] + data['classes'][2:]
        self.words_list = words
        self.word_to_index = {w: i for i, w in enumerate(words)}
        self.data_index = {}
        fallback = next((p for p, y in data['rows']['train'] if p))
        for split, mode in [('train', 'training'), ('validation', 'validation'), ('test', 'testing')]:
            self.data_index[mode] = [dict(label=words[y],
                file=str(Path(WF.data_dir) / (p or fallback))) for p,
                y in data['rows'][split]]
    input_data.AudioProcessor.prepare_data_index = index
    WS = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        device_count={'GPU': 0}))
    with WS.as_default():
        WP = input_data.AudioProcessor(WF)
    WN = np

def worker(job):
    split, view, offset, count = job
    WN.random.seed(20261007 + view * 1000000 + offset)
    augment = split == 'train' and view > 0
    mode = {'train': 'training', 'validation': 'validation', 'test': 'testing'}[split]
    x, y = WP.get_data(count,
        offset,
        WF,
        WF.background_frequency if augment else 0.0,
        WF.background_volume if augment else 0.0,
        int(WF.time_shift_ms * WF.sample_rate / 1000) if augment else 0,
        mode,
        WF.resample if augment else 0.0,
        WF.volume_resample if augment else 0.0,
        WS)
    return (job, x.astype('float32'), y.astype('int64'))

def cache(workers):
    import numpy as np
    source_info('kwt1')
    if (CACHE / 'COMPLETE.json').exists():
        cache_metadata()
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    tf = tf_setup()
    from kws_streaming.layers.speech_features import SpeechFeatures
    flags = SimpleNamespace(**config())
    flags.use_spec_augment = 0
    flags.use_spec_cutout = 0
    params = SpeechFeatures.get_params(flags)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    tf.keras.backend.set_session(sess)
    inp = tf.keras.Input(shape=(16000,))
    layer = SpeechFeatures(params)
    feature_model = tf.keras.Model(inp, layer(inp))
    feature_model._make_predict_function()
    sess.run(tf.global_variables_initializer())
    data = rows()
    files = {}
    jobs = []
    shapes = {}
    start = time.time()
    for split, items in data['rows'].items():
        np.save(CACHE / f'{split}_labels.npy', np.array([y for p, y in items], dtype='int64'))
        for view in range(4 if split == 'train' else 1):
            name = f'{split}_v{view}.npy'
            shape = (len(items), 98, 40)
            files[split,
                view] = np.lib.format.open_memmap(CACHE / name,
                mode='w+',
                dtype='float32',
                shape=shape)
            shapes[name] = list(shape)
            jobs.extend(((split, view, i, min(256, len(items) - i)) for i in range(0, len(items), 256)))
    with ProcessPoolExecutor(max_workers=workers,
        mp_context=mp.get_context('spawn'),
        initializer=worker_init) as pool:
        pending = []
        it = iter(jobs)
        for _ in range(workers * 2):
            j = next(it, None)
            if j is not None:
                pending.append(pool.submit(worker, j))
        done = 0
        while pending:
            job, wave, y = pending.pop(0).result()
            split, view, offset, count = job
            feat = feature_model.predict_on_batch(wave)
            files[split, view][offset:offset + count] = feat
            done += 1
            if done % 10 == 0 or done == len(jobs):
                elapsed = time.time() - start
                atomic(CACHE / 'status.json',
                    dict(status='BUILDING',
                    workers=workers,
                    completed=done,
                    total=len(jobs),
                    elapsed_seconds=elapsed,
                    eta_seconds=elapsed / done * (len(jobs) - done)))
                print('CACHE', done, '/', len(jobs), 'seconds', round(elapsed, 1), flush=True)
            j = next(it, None)
            if j is not None:
                pending.append(pool.submit(worker, j))
    for f in files.values():
        f.flush()
    atomic(CACHE/'COMPLETE.json',dict(status='COMPLETE',shapes=shapes,
        frontend='Author MFCC; one clean and three waveform-augmented training views'))
    atomic(CACHE / 'status.json', dict(status='COMPLETE', elapsed_seconds=time.time() - start))
    sess.close()
