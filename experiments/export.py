'Prepare author features and convert native TensorFlow checkpoints to PyTorch.'
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['bc8', 'tc14', 'kwt1', 'kwmlp'], required=True)
    p.add_argument('--checkpoint',
        type=Path,
        help='Native TensorFlow checkpoint prefix; never a result directory')
    p.add_argument('--config', type=Path, help='Training config.json with args (TC14) or flags (KWT)')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['all', 'native', 'convert'], default='all')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=128)
    return p

def native(a):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from shared.data import load_manifest, read_wave
    from shared.paths import data_root
    a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(a.config.read_text(encoding='utf-8'))
    manifest = load_manifest(a.manifest, a.model)
    root = data_root(a.data_root)
    sys.path[:0] = [str(ROOT / 'experiments/train/_author'), str(ROOT / 'third_party' / a.model)]
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from experiment_common import source_info
    source = source_info(a.model)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as session:
        tf.keras.backend.set_session(session)
        if a.model == 'tc14':
            import tf_keras.src.legacy_tf_layers.normalization as legacy_bn
            import tf_keras.src.legacy_tf_layers as legacy_layers
            sys.modules['tf_keras.legacy_tf_layers.normalization'] = legacy_bn
            sys.modules['tf_keras.legacy_tf_layers'] = legacy_layers
            import factory.audio_nets as nets
            args = SimpleNamespace(**config['args'])
            wave = tf.placeholder(tf.float32, [None, 16000, 1])
            labels = tf.placeholder(tf.float32, [None, 12])
            model = nets.TCResNet14Model(args, None)
            model.build(wave, labels, False)
            session.run(tf.global_variables_initializer())
            tf.train.Saver().restore(session, str(a.checkpoint))
            arrays = {v.op.name: session.run(v) for v in tf.global_variables()}

            def run(batch):
                x, z = session.run([model.audio, model.logits], {wave: batch[..., None]})
                return (x[:, :, :, 0].transpose(0, 2, 1), z)
        else:
            from kws_streaming.models import models, kws_transformer
            from kws_streaming.layers.speech_features import SpeechFeatures
            flags = SimpleNamespace(**config['flags'])
            flags.preprocess = 'mfcc'
            flags.training = False
            flags.use_spec_augment = 0
            flags.use_spec_cutout = 0
            model = kws_transformer.model(flags)
            model._make_predict_function()
            inp = tf.keras.Input(shape=(16000,))
            feature_model = tf.keras.Model(inp, SpeechFeatures(SpeechFeatures.get_params(flags))(inp))
            feature_model._make_predict_function()
            session.run(tf.global_variables_initializer())
            model.load_weights(str(a.checkpoint)).assert_existing_objects_matched()
            transformer = next((layer for layer in model.layers if hasattr(layer, 'enc_layers')))
            arrays = {'pos': session.run(transformer.pos_emb), 'cls': session.run(transformer.class_emb)}

            def dense(prefix, layer):
                values = layer.get_weights()
                arrays[prefix + '.weight'] = values[0].T
                if len(values) > 1:
                    arrays[prefix + '.bias'] = values[1]
            dense('patch', transformer.patch_proj)
            dense('head', model.layers[-1].layers[0])
            for i, block in enumerate(transformer.enc_layers):
                for name, layer in [('q',
                    block.att.query_dense),
                    ('k',
                    block.att.key_dense),
                    ('v',
                    block.att.value_dense),
                    ('proj',
                    block.att.combine_heads),
                    ('ff1',
                    block.ffn.layers[0]),
                    ('ff2',
                    block.ffn.layers[2])]:
                    dense(f'blocks.{i}.{name}', layer)
                for name, layer in [('ln1', block.layernorm1), ('ln2', block.layernorm2)]:
                    w, b = layer.get_weights()
                    arrays[f'blocks.{i}.{name}.weight'] = w
                    arrays[f'blocks.{i}.{name}.bias'] = b

            def run(batch):
                x = np.asarray(feature_model.predict_on_batch(batch))
                return (x, np.asarray(model.predict_on_batch(x)))
        with ThreadPoolExecutor(max_workers=max(a.workers, 1)) as pool:
            for split, rows in manifest['rows'].items():
                target = None
                native_logits = []
                for i in range(0, len(rows), a.batch_size):
                    wave_batch = np.stack(list(pool.map(lambda row: read_wave(root,
                        row),
                        rows[i:i + a.batch_size])))
                    x, z = run(wave_batch)
                    if target is None:
                        target = np.lib.format.open_memmap(a.output / f'{split}_x.npy',
                            mode='w+',
                            dtype='float32',
                            shape=(len(rows),
                            *x.shape[1:]))
                    target[i:i + len(x)] = x
                    if split == 'test':
                        native_logits.append(z)
                target.flush()
                np.save(a.output / f'{split}_y.npy', np.asarray([r[1] for r in rows], np.int64))
                if native_logits:
                    np.save(a.output / 'test_native_logits.npy', np.concatenate(native_logits))
                print('PREPARED', split, len(rows), flush=True)
        np.savez(a.output / 'weights.npz', **arrays)
    (a.output / 'rows.json').write_text(json.dumps(manifest['rows']), encoding='utf-8')
    exported=dict(model=a.model,manifest=manifest,native_frontend=True)
    (a.output / 'native_export.json').write_text(json.dumps(exported, indent=2), encoding='utf-8')

def convert(a):
    import numpy as np
    import torch
    from shared.modeling import create_model, logits
    from shared.data import save_prepared_metadata
    exported = json.loads((a.output / 'native_export.json').read_text(encoding='utf-8'))
    arrays = np.load(a.output / 'weights.npz')
    model = create_model(a.model).float().eval()
    state = model.state_dict()
    if a.model == 'tc14':

        def conv(prefix, tf_prefix):
            state[prefix + '.0.conv.weight'] = torch.from_numpy(arrays[tf_prefix + '/weights'].squeeze(1).transpose(2,
                1,
                0).copy())
            for dst, src in [('weight',
                'gamma'),
                ('bias',
                'beta'),
                ('running_mean',
                'moving_mean'),
                ('running_var',
                'moving_variance')]:
                state[prefix + '.1.' + dst] = torch.from_numpy(arrays[tf_prefix + '/BatchNorm/' + src].copy())
        conv('stem', 'TCResNet14/conv0')
        for i in range(6):
            conv(f'blocks.{i}.conv1', f'TCResNet14/block{i}/conv{i}_0')
            conv(f'blocks.{i}.conv2', f'TCResNet14/block{i}/conv{i}_1')
            if i % 2 == 0:
                conv(f'blocks.{i}.shortcut', f'TCResNet14/block{i}/down')
        state['classifier.weight'] = torch.from_numpy(arrays['TCResNet14/fc/weights'].squeeze((0,
            1)).T.copy())
    else:
        state = {name: torch.from_numpy(arrays[name].copy()) for name in state}
    model.load_state_dict(state, strict=True)
    torch.save(model.state_dict(), a.output / 'model.pth')
    save_prepared_metadata(a.output,
        a.model,
        exported['manifest'],
        'Author-native TensorFlow MFCC',
        dict(native_frontend=True))
    print('COMPLETE', str(a.output.resolve()), flush=True)

def main():
    a = parser().parse_args()
    if a.model in ('bc8', 'kwmlp'):
        from shared.data import prepare_torch
        from shared.paths import data_root
        prepare_torch(a.model, data_root(a.data_root), a.output, a.manifest, a.workers, a.batch_size)
    else:
        if a.stage in ('all', 'native'):
            native(a)
        if a.stage in ('all', 'convert'):
            convert(a)
if __name__ == '__main__':
    main()
