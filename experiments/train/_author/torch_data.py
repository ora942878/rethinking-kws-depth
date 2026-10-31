"""Waveform rows and isolated module loading used by the Torch pair wrapper."""
import importlib.util
import soundfile as sf
import torch
from torch.utils.data import Dataset
from experiment_common import dataset_root

class AudioRows(Dataset):

    def __init__(self, rows, padding):
        self.rows = rows
        self.padding = padding

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, y = self.rows[i]
        if rel is None:
            x = torch.zeros(1, 16000)
        else:
            x, sr = sf.read(dataset_root() / rel, dtype='float32')
            x = torch.from_numpy(x).unsqueeze(0)
        return (self.padding(x), y)

def load_entry(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
