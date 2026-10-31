"""Hyperconnect's native TensorFlow graph, optimizer, augmentation and trainer."""
import copy
import os
import sys
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
from experiment_common import ROOT, atomic, dataset_root, manifest

def run(*, workers, monitor_factory):
    """Run the native author trainer with the caller's paired-run monitor."""
    mon = monitor_factory('tc14', 30000, 36923 / 100)
    sys.path.insert(0, str(ROOT / 'third_party/tc14'))
    import numpy as np
    import random
    random.seed(20261007)
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tf.set_random_seed(20261007)
    np.random.seed(20261007)
    import tf_keras.src.legacy_tf_layers.normalization as legacy_bn
    sys.modules['tf_keras.legacy_tf_layers.normalization'] = legacy_bn
    import tf_keras.src.legacy_tf_layers as legacy_layers
    sys.modules['tf_keras.legacy_tf_layers'] = legacy_layers
    import train_audio
    import const
    import factory.audio_nets as nets
    from datasets.audio_data_wrapper import SingleLabelAudioDataWrapper
    from datasets.augmentation_factory import get_audio_augmentation_fn
    from helper.trainer import TrainerBase
    source = manifest('tc14')
    const.NULL_CLASS_LABEL = source['classes'][0]

    def setup(ds):
        split = {'train': 'train', 'valid': 'validation', 'test': 'test'}[ds.dataset_split_name]
        rows = source['rows'][split]
        ds.label_names = source['classes']
        ds.num_labels = len(ds.label_names)
        ds.file_format = 'wav'
        ds.filenames = [str(dataset_root() / rel) if rel else '' for rel, y in rows]
        ds.labels = [y for rel, y in rows]
        ds.background_max_volume = tf.constant(ds.args.background_max_volume)
        ds.background_frequency = tf.constant(ds.args.background_frequency)
        read = get_audio_augmentation_fn('no_augmentation_audio')
        ds.background_data = [read(tf.constant(str(f)),
            -1,
            'wav',
            16000) for f in sorted((dataset_root() / '_background_noise_').glob('*.wav'))]
        ds._num_samples = len(rows)
        ds.data = (ds.filenames, ds.labels)
        ds.placeholders = (tf.placeholder(tf.string, len(rows)), tf.placeholder(tf.int32, len(rows)))
        if ds.shuffle:
            ds.data = ds.do_shuffle(*ds.data)
    SingleLabelAudioDataWrapper.setup = setup
    args = train_audio.parse_arguments(['--dataset_path',
        str(dataset_root()),
        '--dataset_split_name',
        'train',
        '--output_name',
        'output/softmax',
        '--num_classes',
        '12',
        '--train_dir',
        str(mon.weights),
        '--num_silent',
        '0',
        '--augmentation_method',
        'anchored_slice_or_pad_with_shift',
        '--preprocess_method',
        'mfcc',
        '--num_mfccs',
        '40',
        '--clip_duration_ms',
        '1000',
        '--window_size_ms',
        '30',
        '--window_stride_ms',
        '10',
        '--batch_size',
        '100',
        '--boundaries',
        '10000',
        '20000',
        '--max_step_from_restore',
        str(mon.total),
        '--lr_list',
        '0.1',
        '0.01',
        '0.001',
        '--absolute_schedule',
        '--no-boundaries_epoch',
        '--max_to_keep',
        '3',
        '--step_save_checkpoint',
        '500',
        '--step_evaluation',
        '500',
        '--step_save_first_n_summaries',
        '0',
        '--no-write_pbtxt',
        '--optimizer',
        'mom',
        '--momentum',
        '0.9',
        'TCResNet14Model',
        '--weight_decay',
        '0.001',
        '--width_multiplier',
        '1.0'])
    args.num_threads = workers
    atomic(mon.out / 'config.json',
        dict(args=vars(args),
        source=mon.values['source'],
        manifest='configs/manifests/gsc12.json',
        initialization='author random initialization'))

    class Evaluation:

        def __init__(self, split, training_session):
            self.graph = tf.Graph()
            with self.graph.as_default():
                cfg = copy.copy(args)
                cfg.dataset_split_name = [split]
                cfg.shuffle = False
                cfg.augmentation_method = 'anchored_slice_or_pad'
                cfg.background_frequency = 0.0
                cfg.background_max_volume = 0.0
                self.session = tf.Session(config=const.TF_SESSION_CONFIG)
                self.ds = SingleLabelAudioDataWrapper(cfg, self.session, split, False)
                wav, y = self.ds.get_input_and_output_op()
                self.y = y
                self.model = nets.TCResNet14Model(cfg, self.ds)
                self.model.build(wavs=wav, labels=y, is_training=False)
                self.pred = tf.argmax(self.model.logits, axis=1)
                self.truth = tf.argmax(y, axis=1)
                self.variables = tf.global_variables()
                self.inputs = [tf.placeholder(v.dtype, v.shape) for v in self.variables]
                self.assign = [tf.assign(v, x) for v, x in zip(self.variables, self.inputs)]
                self.session.run(tf.global_variables_initializer())
            self.training_session = training_session

        def evaluate(self, trainer):
            with trainer.session.graph.as_default():
                train_vars = {v.name: v for v in tf.global_variables()}
                tensors = trainer.session.run([train_vars[v.name] for v in self.variables])
            self.session.run(self.assign, dict(zip(self.inputs, tensors)))
            self.session.run(self.ds.iterator.initializer, dict(zip(self.ds.placeholders, self.ds.data)))
            correct = 0
            total = 0
            while True:
                try:
                    y, p = self.session.run([self.truth, self.pred])
                    correct += int((y == p).sum())
                    total += len(y)
                except tf.errors.OutOfRangeError:
                    break
            return 100 * correct / total
    eval_state = {}
    orig = TrainerBase.run_with_logging

    def run_step(trainer, *args_, **kw):
        result = orig(trainer, *args_, **kw)
        mon.step = int(trainer.session.run(trainer.global_step))
        if mon.step % 25 == 0:
            vals = result[0]
            mon.record('STEP',
                total_loss=float(vals[trainer.loss_fetch_namespace.total_loss]),
                model_loss=float(vals[trainer.loss_fetch_namespace.model_loss]),
                lambda_value=mon.weight())
        if mon.step % 500 == 0 or mon.step >= mon.total:
            if 'validation' not in eval_state:
                eval_state['validation'] = Evaluation('valid', trainer.session)
            acc = eval_state['validation'].evaluate(trainer)
            if acc > mon.best:
                if 'best_saver' not in eval_state:
                    with trainer.session.graph.as_default():
                        eval_state['best_saver'] = tf.train.Saver(max_to_keep=1)
                mon.best = acc
                best_dir = mon.weights / 'best'
                best_dir.mkdir(exist_ok=True)
                eval_state['best_path'] = eval_state['best_saver'].save(trainer.session,
                    str(best_dir / 'model'),
                    latest_filename='best_checkpoint')
                mon.record('BEST_SAVED',
                    checkpoint=eval_state['best_path'],
                    best_step=mon.step,
                    best_validation_accuracy=acc)
            mon.record('VALIDATION', validation_accuracy=acc, best_validation_accuracy=mon.best)
        eval_state['trainer'] = trainer
        return (result[0],
            mon.step,
            mon.step - trainer.global_step_from_checkpoint,
            (mon.step - trainer.global_step_from_checkpoint) / (36923 / 100))
    TrainerBase.run_with_logging = run_step
    train_audio.train(args)
    trainer = eval_state['trainer']
    with trainer.session.graph.as_default():
        final_saver = tf.train.Saver(max_to_keep=1)
        variables = tf.global_variables()
    last_dir = mon.weights / 'last'
    last_dir.mkdir(exist_ok=True)
    last_path = final_saver.save(trainer.session, str(last_dir / 'model'), latest_filename='last_checkpoint')
    eval_state['best_saver'].restore(trainer.session, eval_state['best_path'])
    test = Evaluation('test', trainer.session)
    acc = test.evaluate(trainer)
    mon.record('BEST_TEST', best_test_accuracy=acc)
    test.session.close()
    mon.finish(author_loop='tc14/train_audio.py and helper/trainer.py')
