"""Causal features from frozen Backbone for a compact optional correction."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .compact_high import CompactLocalSNN
from .model import T
from .select_controllers import load_model

ROOT = Path('results/compact_high_candidate_v2')

@torch.no_grad()
def extract(split, device):
    ROOT.mkdir(parents=True, exist_ok=True)
    dest = ROOT / f'{split}_features.npz'
    assert not dest.exists()
    if split == 'fit':
        old = np.load('results/unified_followup/fit_cache.npz')
        ids = old['indices']
        frames = np.load('data/prepared/train_frames.npy', mmap_mode='r')
    elif split == 'validation':
        old = np.load('results/local_seed42/controllers/validation_cache.npz')
        ids = old['indices']
        frames = np.load('data/prepared/train_frames.npy', mmap_mode='r')
    else:
        old = np.load('results/local_seed42/controllers/final/test_cache.npz')
        ids = np.arange(len(old['low']))
        frames = np.load('data/prepared/test_frames.npy', mmap_mode='r')
    model, _ = load_model('results/refine_adapt_subject_seed42/best.pt', device)
    feats = []
    max_low_diff = max_rate_diff = 0.
    for start in range(0, len(ids), 16):
        x = torch.as_tensor(np.array(frames[ids[start:start+16]], copy=True), device=device)
        lows = []
        rates = []
        f = []
        for t in range(T):
            low, local = model.step(x[:, t])
            lows.append(low)
            rates.append(model.activity(local))
            partial = torch.stack(lows, 1).mean(1)
            f.append(CompactLocalSNN.compact_features(local, partial).cpu().numpy())
        low = torch.stack(lows, 1).mean(1).cpu().numpy()
        rate = torch.stack(rates, 1).cpu().numpy()
        n = len(x)
        max_low_diff = max(max_low_diff, float(np.max(np.abs(low - old['low'][start:start+n]))))
        max_rate_diff = max(max_rate_diff, float(np.max(np.abs(rate - old['rate'][start:start+n]))))
        feats.append(np.stack(f, 1))
        model.reset()
        if start % 160 == 0:
            print(split, start+n, '/', len(ids), flush=True)
    assert max_low_diff < 2e-5 and max_rate_diff < 2e-5, (max_low_diff, max_rate_diff)
    np.savez_compressed(dest, features=np.concatenate(feats).astype(np.float32),
                        indices=ids, low=old['low'], rate=old['rate'],
                        labels=old['labels'] if 'labels' in old.files else np.load('data/prepared/test_labels.npy'))
    print('SAVED', dest, 'max diff', max_low_diff, max_rate_diff, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=('fit','validation','official_test'), required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    extract(a.split, a.device)
