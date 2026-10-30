"""Matched two-paper adapted scheduler evaluation on frozen Compact High."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .compact_high import CompactHigh, compact_extra_mac
from .model import T, dense_mac_per_t
from .outage_sweep import stream_inputs
from .predictive_online import predict
from .recent_online_scheduler import external_simulate
from .seven_baselines import SEEDS, CKPATH, CKSHA

ROOT = Path('results/recent_online_schedulers_2025')
FEATURE_ROOT = Path('results/compact_high_candidate_v2')
CHECKPOINT = FEATURE_ROOT / 'best.pt'
METHODS = ('dsm_2025_adapted', 'ecss_2025_adapted')
PAPERS = {
    'dsm_2025_adapted': 'https://arxiv.org/abs/2510.27317',
    'ecss_2025_adapted': 'https://link.springer.com/article/10.1186/s13677-025-00780-7',
}
GRID = {'dsm_2025_adapted': (0.0, 0.5, 1.0),
        'ecss_2025_adapted': (0.25, 0.5, 1.0)}


def save_csv(path, rows):
    assert rows and not path.exists(), path
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def load_split(split, head):
    f = np.load(FEATURE_ROOT / f'{split}_features.npz')
    rate = f['rate']; low = f['low']; labels = f['labels']
    x = torch.as_tensor(f['features'], dtype=torch.float32)
    with torch.no_grad():
        corr = head(x.reshape(-1, x.shape[-1])).reshape(len(x), T, 4, 11).numpy()
    assert corr.shape == (len(labels), T, 4, 11)
    return rate, low, labels, corr


def evaluate_one(inp, forecast, low, labels, corr, method, param):
    mask, st, _ = external_simulate(inp, forecast, method, param)
    logits = low + np.einsum('ntg,ntgc->nc', mask, corr, optimize=True) / T
    base, _ = dense_mac_per_t()
    return {'accuracy': float(np.mean(logits.argmax(1) == labels)),
            'executed_mac_per_clip': T * base + st['high_calls_per_clip'] * compact_extra_mac(),
            **st}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'official_test_summary.csv').exists(), 'Refuse overwrite'
    ck_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert ck_sha == '717df1cdc60c13e93a572d22f542df7f7df25ecb3637b6798e85872fc9c50de2'
    assert hashlib.sha256(CKPATH.read_bytes()).hexdigest() == CKSHA
    torch.set_num_threads(4)
    head = CompactHigh().eval()
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    head.load_state_dict(ck['compact'])
    coef = np.asarray(json.loads(Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    val_rate, val_low, val_labels, val_corr = load_split('validation', head)
    val_traces = [(seed, stream_inputs(val_rate, 'd64_v50', seed)) for seed in SEEDS['validation']]
    fit_rows = []
    chosen = {}
    # Resource-only calibration. No validation class labels, and no test data, enter selection.
    for method in METHODS:
        for param in GRID[method]:
            stats = []
            for seed, inp in val_traces[:2]:
                _, st, _ = external_simulate(inp, predict(inp, coef), method, param)
                stats.append(st)
            rcir = float(np.mean([s['rcir'] for s in stats]))
            outage = float(np.mean([s['outage_service'] for s in stats]))
            fit_rows.append(dict(method=method, parameter=param, calibration_seeds='1201,1202',
                                 rcir=rcir, outage_service=outage,
                                 objective=0.5 * rcir - 0.5 * outage))
        chosen[method] = min((r for r in fit_rows if r['method'] == method),
                             key=lambda r: (r['objective'], r['parameter']))['parameter']
    save_csv(ROOT / 'validation_calibration.csv', fit_rows)
    print('CHOSEN', chosen, flush=True)
    all_rows = []
    for split in ('validation', 'official_test'):
        if split == 'validation':
            rate, low, labels, corr = val_rate, val_low, val_labels, val_corr
            traces = val_traces
        else:
            rate, low, labels, corr = load_split(split, head)
            traces = [(seed, stream_inputs(rate, 'd64_v50', seed)) for seed in SEEDS[split]]
        rows = []
        for seed, inp in traces:
            forecast = predict(inp, coef)
            hashes = None
            for method in METHODS:
                result = evaluate_one(inp, forecast, low, labels, corr, method, chosen[method])
                pair = result['original_requests_hash'], result['supply_hash']
                assert hashes is None or pair == hashes
                hashes = pair
                assert result['requests'] == 32 * len(labels)
                rows.append(dict(split=split, seed=seed, method=method,
                                 parameter=chosen[method], checkpoint_sha256=ck_sha,
                                 **result))
            print('EVALUATED', split, seed, flush=True)
        save_csv(ROOT / f'{split}_raw.csv', rows)
        keys = ('accuracy', 'rcir', 'outage_service', 'executed_mac_per_clip',
                'high_calls_per_clip', 'spill_tokens', 'transferred_tokens')
        summary = [dict(split=split, method=method, nclips=len(labels),
                        n_supply_seeds=len(traces), checkpoint_sha256=ck_sha,
                        parameter=chosen[method], paper=PAPERS[method],
                        **{k: float(np.mean([r[k] for r in rows if r['method'] == method])) for k in keys})
                   for method in METHODS]
        save_csv(ROOT / f'{split}_summary.csv', summary)
        all_rows.extend(summary)
        print('SUMMARY', split, json.dumps(summary), flush=True)
    save_csv(ROOT / 'two_paper_table.csv', all_rows)
    (ROOT / 'manifest.json').write_text(json.dumps(dict(
        dataset='DVS128 Gesture', scenario='d64_v50', checkpoint=str(CHECKPOINT),
        backbone_checkpoint=str(CKPATH), backbone_sha256=CKSHA,
        checkpoint_sha256=ck_sha, source_git_commit='ef24896',
        current_git_branch='recent-online-scheduler-baselines-2025',
        supply_seeds=SEEDS, original_high_requests_per_clip=32,
        reserve_capacity_per_region=4, initial_reserve_per_region=2,
        supply_allocation_delay_bins=4, adjacent_transfer_budget_per_bin=1,
        methods=PAPERS, chosen_parameters=chosen,
        calibration='Validation supply seeds 1201,1202; minimize 0.5*RCIR-0.5*outage_service; no class labels',
        caveat='Adaptations to SNN token sharing, not native paper reproductions. DSM DVFS and ECSS external nodes are unavailable.',
        test_labels_used_for_tuning=False, pirc_not_rerun=True), indent=2))

if __name__ == '__main__': main()
