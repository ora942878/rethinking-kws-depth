"""Ordered Speech Commands manifests and explicit unaugmented feature caches."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]



def load_manifest(path=None, kind='bc8'):
    path = Path(path) if path else ROOT / 'configs/manifests' / ('gsc35.json' if kind == 'kwmlp' else 'gsc12.json')
    return json.loads(path.read_text(encoding='utf-8'))

def load_prepared(path, device='cpu', manifest_path=None):
    import torch
    path = Path(path)
    return {split: (torch.from_numpy(np.load(path / f'{split}_x.npy')).float().to(device),
        torch.from_numpy(np.load(path / f'{split}_y.npy')).long().to(device)) for split in ('train',
        'validation',
        'test')}

def read_wave(root, row):
    import soundfile as sf
    wave = np.zeros(16000, np.float32)
    if row[0] is not None:
        value, sr = sf.read(Path(root) / row[0], dtype='float32')
        wave[:len(value)] = value
    return wave

def save_prepared_metadata(out,kind,manifest,frontend,extra=None):
    out=Path(out)
    metadata=dict(model=kind,classes=manifest['classes'],frontend=frontend,augmentation=False)
    metadata.update(extra or {})
    (out/'rows.json').write_text(json.dumps(manifest['rows']),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

def prepare_torch(kind, data_root, out, manifest_path=None, workers=4, batch_size=128):
    """Prepare complete fixed splits with the authors' BC8/KW-MLP frontends."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    manifest = load_manifest(manifest_path, kind)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if kind == 'bc8':
        import torchaudio
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
            n_fft=512,
            win_length=480,
            hop_length=160,
            n_mels=40)

        def features(row):
            return read_wave(data_root, row)
        frontend = 'torchaudio log-Mel: 16000 Hz, FFT512, window480, hop160, 40 bands, log(mel+1e-6)'
    else:
        import librosa

        def features(row):
            wave = read_wave(data_root, row)
            mel = librosa.feature.melspectrogram(y=wave,
                sr=16000,
                n_mels=40,
                n_fft=480,
                win_length=480,
                hop_length=160,
                center=False)
            return librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=40)[None].astype(np.float32)
        frontend = 'librosa MFCC: 16000 Hz, FFT480, window480, hop160, center=False, 40 coefficients'
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for split, rows in manifest['rows'].items():
            target = None
            for start in range(0, len(rows), batch_size):
                x = np.stack(list(pool.map(features, rows[start:start + batch_size])))
                if kind == 'bc8':
                    with torch.inference_mode():
                        x = (transform(torch.from_numpy(x).unsqueeze(1)) + 1e-06).log().numpy()
                if target is None:
                    target = np.lib.format.open_memmap(out / f'{split}_x.npy',
                        mode='w+',
                        dtype='float32',
                        shape=(len(rows),
                        *x.shape[1:]))
                target[start:start + len(x)] = x
            if target is not None:
                target.flush()
            np.save(out / f'{split}_y.npy', np.asarray([r[1] for r in rows], np.int64))
    save_prepared_metadata(out, kind, manifest, frontend)
    return out
