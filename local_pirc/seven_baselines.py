"""Matched seven-method DVS resource evaluation with one frozen High head."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .outage_sweep import stream_inputs
from .predictive_online import predict
from .unified_system import simulate

ROOT = Path('results/seven_baseline_head_v3')
CKPATH = Path('results/refine_adapt_subject_seed42/best.pt')
METHODS = ('fixed_allocation', 'instant_gating', 'ema_gating',
           'reserve_threshold', 'balanced_sharing', 'predictive_reserve', 'pirc')
SEEDS = {'validation': (1201, 1202, 1203, 1204, 1205),
         'official_test': (1301, 1302, 1303, 1304, 1305)}


def save(path, rows):
    assert rows and not path.exists(), f'Refuse overwrite {path}'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def load_cache(split):
    if split == 'validation':
        old = np.load('results/local_seed42/controllers/validation_cache.npz')
        labels = old['labels']
    elif split == 'official_test':
        old = np.load('results/local_seed42/controllers/final/test_cache.npz')
        labels = np.load('data/prepared/test_labels.npy')
    else:
        raise ValueError(split)
    feat = np.load(f'results/refine_adapt_seed42/{split}_features.npz')['features']
    ck = torch.load(CKPATH, map_location='cpu', weights_only=False)['model']
    corr = np.einsum('ntgf,cf->ntgc', feat, ck['refine.head.weight'].numpy(), optimize=True)
    corr += ck['refine.head.bias'].numpy()
    assert corr.shape == old['corr'].shape
    assert len(labels) == len(old['low']) == len(old['rate'])
    return {'low': old['low'], 'corr': corr, 'rate': old['rate'], 'labels': labels}


def run_policy(inp, forecast, method, cfg):
    if method == 'fixed_allocation':
        return simulate(inp, forecast, 'traditional_share', cfg,
                        share_mode='none', allocation_mode='fixed')
    if method == 'instant_gating':
        return simulate(inp, forecast, 'traditional_share', cfg,
                        share_mode='none', allocation_mode='instant')
    if method == 'ema_gating':
        return simulate(inp, forecast, 'traditional_share', cfg,
                        share_mode='none', allocation_mode='ema')
    if method == 'reserve_threshold':
        return simulate(inp, forecast, 'traditional_share', cfg)
    if method == 'balanced_sharing':
        return simulate(inp, forecast, 'traditional_share', cfg, share_mode='balance')
    if method == 'predictive_reserve':
        return simulate(inp, forecast, 'predictive_share', cfg)
    if method == 'pirc':
        return simulate(inp, forecast, 'pirc_full', cfg)
    raise ValueError(method)


def score(cache, mask):
    logits = cache['low'] + np.einsum('ntg,ntgc->nc', mask, cache['corr'], optimize=True) / 16
    return float(np.mean(logits.argmax(1) == cache['labels']))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cksha = hashlib.sha256(CKPATH.read_bytes()).hexdigest()
    cfg = json.loads(Path('configs/resource_controller.json').read_text())['config']
    coef = np.asarray(json.loads(Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    for split in ('validation', 'official_test'):
        assert not (ROOT / f'{split}_summary.csv').exists(), 'Refuse overwrite'
        cache = load_cache(split)
        rows = []; first_masks = {}
        for seed in SEEDS[split]:
            inp = stream_inputs(cache['rate'], 'd64_v50', seed)
            forecast = predict(inp, coef)
            hashes = None
            for method in METHODS:
                mask, st = run_policy(inp, forecast, method, cfg)
                pair = (st['original_requests_hash'], st['supply_hash'])
                if hashes is None: hashes = pair
                assert pair == hashes
                assert st['requests'] == 32 * len(cache['labels'])
                rows.append({'split': split, 'seed': seed, 'method': method,
                             'accuracy': score(cache, mask), **st})
                if seed == SEEDS[split][0]: first_masks[method] = mask
            print('EVALUATED', split, seed, flush=True)
        save(ROOT / f'{split}_raw.csv', rows)
        np.savez_compressed(ROOT / f'{split}_first_seed_masks.npz', **first_masks)
        keys = ('accuracy', 'rcir', 'outage_service', 'executed_mac_per_clip',
                'high_calls_per_clip', 'overflow_rate', 'transferred_tokens',
                'global_crsr', 'adjacent_crsr', 'preoutage_reserve_per_region')
        summary = []
        for method in METHODS:
            sub = [r for r in rows if r['method'] == method]
            summary.append({'method': method, 'split': split,
                            'nclips': len(cache['labels']), 'n_seeds': len(SEEDS[split]),
                            'checkpoint_sha256': cksha,
                            **{key: float(np.mean([r[key] for r in sub])) for key in keys}})
        save(ROOT / f'{split}_summary.csv', summary)
        print('SUMMARY', split, json.dumps(summary), flush=True)
    (ROOT / 'manifest.json').write_text(json.dumps({
        'checkpoint': str(CKPATH), 'checkpoint_sha256': cksha,
        'dataset': 'DVS128 Gesture', 'scenario': 'd64_v50',
        'capacity': 4, 'initial_reserve': 2, 'allocation_delay': 4,
        'original_requests_per_clip': 32, 'adjacent_transfer_budget_per_bin': 1,
        'methods': METHODS, 'seeds': SEEDS, 'pirc_config': cfg,
        'all_methods_share_checkpoint_requests_and_supply': True,
        'official_test_labels_used_for_tuning': False,
        'both_splits_previously_inspected': True,
    }, indent=2))


if __name__ == '__main__':
    main()
