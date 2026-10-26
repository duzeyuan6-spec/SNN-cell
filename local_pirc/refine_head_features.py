"""Extract the existing frozen refinement-head inputs; verify old output identity."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .model import T, GROUPS
from .select_controllers import load_model

ROOT = Path('results/refine_adapt_seed42')


@torch.no_grad()
def extract(split, device='cuda:0'):
    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / f'{split}_features.npz'
    assert not out.exists(), 'Refuse overwrite'
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
        frames = np.load('data/prepared/test_frames.npy', mmap_mode='r')
        ids = np.arange(len(frames))
    torch.set_num_threads(4)
    model, _ = load_model('results/local_seed42/best.pt', device)
    feats = []
    max_corr_error = max_low_error = max_rate_error = 0.
    for start in range(0, len(ids), 16):
        sel = ids[start:start+16]
        x = torch.as_tensor(np.array(frames[sel], copy=True), device=device)
        batch_feat = []
        batch_low = []
        batch_rate = []
        for t in range(T):
            low, local = model.step(x[:, t])
            patches = model.patches(local)
            region = torch.arange(len(x)*GROUPS, device=device) % GROUPS
            f = model.refine.features(patches, region).reshape(len(x), GROUPS, 72)
            batch_feat.append(f.cpu().numpy())
            batch_low.append(low.cpu().numpy())
            batch_rate.append(model.activity(local).cpu().numpy())
        model.reset()
        f = np.stack(batch_feat, axis=1)
        low = np.stack(batch_low, axis=1).mean(1)
        rate = np.stack(batch_rate, axis=1)
        with torch.no_grad():
            corr = model.refine.head(torch.as_tensor(f, device=device)).cpu().numpy()
        max_corr_error = max(max_corr_error, float(np.max(np.abs(corr-old['corr'][start:start+len(sel)]))))
        max_low_error = max(max_low_error, float(np.max(np.abs(low-old['low'][start:start+len(sel)]))))
        max_rate_error = max(max_rate_error, float(np.max(np.abs(rate-old['rate'][start:start+len(sel)]))))
        feats.append(f)
        print(split, start+len(sel), '/', len(ids), flush=True)
    assert max_corr_error < 2e-5 and max_low_error < 2e-5 and max_rate_error < 2e-5, (max_corr_error,max_low_error,max_rate_error)
    np.savez_compressed(out, features=np.concatenate(feats).astype(np.float32), indices=ids)
    print('SAVED', out, 'max errors', max_corr_error, max_low_error, max_rate_error, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('fit','validation','official_test'), required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    extract(a.split, a.device)


if __name__ == '__main__':
    main()
