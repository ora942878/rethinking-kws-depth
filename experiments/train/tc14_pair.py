'TC-ResNet-14 source training with the Hyperconnect loop and moment loss.'
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
AUTHOR = HERE / '_author'
SEED = 20261007



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lambda-max', type=float, choices=[0.0, 0.1], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    tag = 'lambda_0' if args.lambda_max == 0 else 'lambda_0p1'
    stage = 'full'
    out = ROOT / 'outputs/train' / args.run_id / 'tc14' / tag / stage
    weights = ROOT / 'weights' / args.run_id / 'tc14' / tag / stage
    out.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    os.environ.update(TF_USE_LEGACY_KERAS='1',
        TF_DETERMINISTIC_OPS='1',
        TF_CUDNN_DETERMINISTIC='1',
        TF_ENABLE_ONEDNN_OPTS='0',
        TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_NUM_INTRAOP_THREADS=str(args.workers),
        TF_NUM_INTEROP_THREADS='1',
        OMP_NUM_THREADS='1')
    sys.path.insert(0, str(AUTHOR))
    import experiment_common as common
    atomic = common.atomic
    monitor = [None]
    recipe = dict(model='tc14',seed=SEED,batch_size=100,updates=30000,
        optimizer='SGD',momentum=.9,weight_decay=.001,learning_rates=[.1,.01,.001],
        boundaries=[10000,20000],warmup_epochs=5,ramp_epochs=15)

    class PairMonitor(common.Monitor):

        def __init__(self, kind, total_steps, steps_per_epoch):
            self.kind = kind
            self.total = 30000
            self.spe = steps_per_epoch
            self.out = out
            self.weights = weights
            self.started = time.time()
            self.step = 0
            self.best = -1
            self.values = {}
            monitor[0] = self
            self.update('PREPARING',
                source=common.source_info(kind),
                total_steps=self.total,
                pairs=common.PAIRS[kind],
                lambda_max=args.lambda_max,
                warmup_epochs=5,
                ramp_epochs=15,
                python_executable=sys.executable,
                initialization='author random initialization',
                seed=SEED,
                workers=args.workers,
                variant=tag,
                run_id=args.run_id,
                recipe=recipe,
                output_directory=str(out),
                weights_directory=str(weights))

        def weight(self):
            factor = min(1.0, max(0.0, (self.step / self.spe - 5) / 15))
            return args.lambda_max * factor
    try:
        import tensorflow as tensorflow
        tf = tensorflow.compat.v1
        tf.disable_eager_execution()
        tensorflow.config.experimental.enable_op_determinism()
        tensorflow.config.experimental.enable_tensor_float_32_execution(False)
        original_graph = tf.Graph

        class SeededGraph(original_graph):

            def __init__(self, *pos, **kw):
                super().__init__(*pos, **kw)
                self.seed = SEED
        tf.Graph = SeededGraph
        sys.path.insert(0, str(ROOT / 'third_party/tc14'))
        import tf_moments
        import tc14_adapter as adapter
        from datasets.data_wrapper_base import DataWrapperBase
        from helper.trainer import TrainerBase

        def paired_tc_loss():
            outputs = tf.get_collection('moment_block_outputs')
            step = tf.train.get_or_create_global_step()
            with tf.variable_scope('matched_moment_pair', reuse=tf.AUTO_REUSE):
                coefficient = tf.get_variable('lambda_max',
                    shape=(),
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                    trainable=False)
            ramp = tf.clip_by_value((tf.cast(step, tf.float32) / (36923 / 100) - 5) / 15, 0.0, 1.0)
            raw = tf_moments.regularizer(outputs, [(2, 1), (4, 3), (6, 5)])
            tf.add_to_collection('moment_raw_loss', raw)
            tf.add_to_collection('paired_moment_coefficient', coefficient)
            return coefficient * ramp * raw
        tf_moments.tc_loss = paired_tc_loss

        def deterministic_dataset(ds, placeholders, batch_size=None):
            """Author transformation order with explicit shared shuffle seeds."""
            ds.batch_size = ds.args.batch_size if batch_size is None else batch_size
            data = tf.data.Dataset.from_tensor_slices(placeholders)
            if ds.shuffle:
                data = data.shuffle(ds.num_samples, seed=SEED, reshuffle_each_iteration=True)
            data = data.map(ds._parse_function, num_parallel_calls=args.workers)
            data = data.prefetch(tf.data.experimental.AUTOTUNE)
            data = data.batch(ds.batch_size)
            if ds.is_training and ds.shuffle:
                data = data.shuffle(ds.args.buffer_size,
                    seed=SEED + 1,
                    reshuffle_each_iteration=True).repeat(-1)
            elif ds.is_training:
                data = data.repeat(-1)
            options = tf.data.Options()
            options.experimental_deterministic = True
            options.threading.private_threadpool_size = args.workers
            data = data.with_options(options)
            ds.dataset = data
            ds.iterator = data.make_initializable_iterator()
            ds.next_elem = ds.iterator.get_next()
        DataWrapperBase.setup_dataset = deterministic_dataset
        restore = TrainerBase.routine_restore_and_initialize

        def initialize_coefficient(trainer, *pos, **kw):
            restore(trainer, *pos, **kw)
            with trainer.session.graph.as_default():
                coefficient = tf.get_collection('paired_moment_coefficient')[0]
                assign = tf.assign(coefficient, args.lambda_max)
            trainer.session.run(assign)
        TrainerBase.routine_restore_and_initialize = initialize_coefficient

        def write_config(path, value):
            if Path(path).name == 'config.json':
                value.update(recipe=recipe,
                    lambda_max=args.lambda_max,
                    variant=tag,
                    seed=SEED,
                    only_experimental_variable='lambda_max')
            atomic(path, value)
        adapter.atomic = write_config
        adapter.run(workers=args.workers, monitor_factory=PairMonitor)
    except Exception as error:
        if monitor[0] is not None:
            monitor[0].update('FAILED', error=repr(error), traceback=traceback.format_exc())
        else:
            atomic(out / 'status.json',
                dict(status='FAILED',
                error=repr(error),
                traceback=traceback.format_exc(),
                lambda_max=args.lambda_max,
                seed=SEED,
                model='tc14',
                run_id=args.run_id))
        raise
if __name__ == '__main__':
    main()
