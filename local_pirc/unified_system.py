"""Frozen-weight, matched-request unified resource experiment."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .local_sharing import G, NEIGHBORS, transfer
from .model import T, dense_mac_per_t
from .outage_sweep import DELAY, stream_inputs
from .predictive_online import accuracy, features, predict, target

ROOT = Path('results/unified_system_seed42')
PROTOCOL = '7f1ac39'
SEEDS = tuple(range(731, 741))
SCENARIO = 'd64_v50'
CAP = 4.
INITIAL = 2.
POLICIES = ('traditional_share', 'pirc_full', 'pirc_noreserve',
            'pirc_nohistory', 'pirc_nosharing')


def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


def current_only_coef(rates):
    gram = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for seed in (701, 702, 703):
        inp = stream_inputs(rates, SCENARIO, seed)
        x = features(inp['activity'], inp['requested'])[:, :, (0, 1, 4)].reshape(-1, 3)
        y = target(inp['requested']).reshape(-1)
        gram += x.T @ x
        rhs += x.T @ y
    reg = np.diag((0., .001, .001))
    return np.linalg.solve(gram + reg, rhs)


def current_only_predict(inp, coef):
    x = features(inp['activity'], inp['requested'])[:, :, (0, 1, 4)]
    return np.clip(x @ coef, 0., 1.)


def simulate(inp, forecast, policy, config=None, utility_priority=None, *, share_mode=None, priority_weights=None, state_trace=None, allocation_mode=None):
    assert policy in POLICIES or policy == 'predictive_share'
    assert share_mode in (None, 'balance', 'priority', 'none')
    assert allocation_mode in (None, 'fixed', 'instant', 'ema')
    if allocation_mode is not None: assert policy == 'traditional_share' and share_mode == 'none'
    if share_mode == 'none': assert policy == 'traditional_share'
    if share_mode == 'balance': assert policy == 'traditional_share'
    if share_mode == 'priority':
        assert policy == 'pirc_full' and utility_priority is None
        assert priority_weights is not None and len(priority_weights) == 3
        assert all(float(w) >= 0. for w in priority_weights)
    cfg = {'rho': .95, 'safe_floor': 0., 'safe_gap_gain': 4.,
           'safe_forecast_gain': 1., 'reserve_alloc_gain': 2.,
           'alloc_forecast_gain': 1., 'burst_gain': 0.,
           'burst_threshold': 2., 'burst_ema_rho': .9,
           'transfer_priority_gain': 0., 'lookahead_gain': 0.,
           'fast_gain': 4., 'burst_alloc_gain': 0., 'burst_projected_gain': 0.}
    if config is not None: cfg.update(config)
    n = len(inp['supply'])
    assert forecast.shape == (n, G)
    if utility_priority is not None:
        assert utility_priority.shape == (n, G) and np.isfinite(utility_priority).all()
    if state_trace is not None:
        assert isinstance(state_trace, dict) and not state_trace
        state_trace['available_pretransfer'] = np.empty((n, G), dtype=np.float32)
        state_trace['queued_supply'] = np.empty((n, G), dtype=np.float32)
        state_trace['slow_supply'] = np.empty((n, G), dtype=np.float32)
        state_trace['supply_ema'] = np.empty(n, dtype=np.float32)
    req = inp['requested']
    supply = inp['supply']
    reserve = np.full(G, INITIAL, dtype=np.float64)
    queue = [np.full(G, 1. / G) for _ in range(DELAY)]
    slow_req = np.zeros(G, dtype=np.float64)
    slow_activity = np.zeros(G, dtype=np.float64)
    slow_supply = np.zeros(G, dtype=np.float64)
    supply_ema = 1.6
    served = np.zeros_like(req)
    spill = 0.
    global_crsr = adjacent_crsr = 0
    transferred = 0.
    competing_recipients = 0
    priority_order_changes = 0
    safe_total = 0.
    preoutage_reserve = []
    transfer_by_phase = {'normal': 0., 'burst': 0., 'outage': 0.}
    for t in range(n):
        incoming = supply[t] * queue.pop(0)
        pre = reserve + incoming
        spill += np.maximum(pre - CAP, 0.).sum()
        available = np.minimum(pre, CAP)
        q = req[t].astype(float)
        supply_ema = cfg['burst_ema_rho'] * supply_ema + (1. - cfg['burst_ema_rho']) * supply[t]
        rho = cfg['rho']
        slow_req = rho * slow_req + (1. - rho) * q
        slow_activity = .9 * slow_activity + .1 * inp['activity'][t]
        slow_supply = rho * slow_supply + (1. - rho) * incoming
        if state_trace is not None:
            state_trace['available_pretransfer'][t] = available
            state_trace['queued_supply'][t] = supply_ema * np.sum(queue, axis=0)
            state_trace['slow_supply'][t] = slow_supply
            state_trace['supply_ema'][t] = supply_ema
        if policy in ('traditional_share', 'predictive_share'):
            safe = np.ones(G, dtype=np.float64)
        elif policy == 'pirc_noreserve':
            safe = np.zeros(G, dtype=np.float64)
        else:
            safe = np.clip(cfg['safe_floor'] +
                           cfg['safe_gap_gain'] * np.maximum(slow_req - slow_supply, 0.) +
                           cfg['safe_forecast_gain'] * np.maximum(forecast[t] - q, 0.) +
                           cfg['burst_gain'] * max(supply_ema - cfg['burst_threshold'], 0.),
                           0., CAP)
        safe_total += float((np.zeros(G) if share_mode in ('balance', 'none') else safe).sum())
        deficient = np.flatnonzero(req[t] & (available < 1. - 1e-9))
        if len(deficient) > 1: competing_recipients += 1
        local_priority = utility_priority[t] if utility_priority is not None else None
        if share_mode == 'priority':
            wr, wres, whist = priority_weights
            pending = supply_ema * np.sum(queue, axis=0)
            risk = np.maximum(0., DELAY * forecast[t] - available - pending) / CAP
            local_priority = wr * risk + wres * (CAP - available) / CAP + whist * slow_req
            if len(deficient) > 1:
                ordered = sorted(deficient, key=lambda r: (-local_priority[r], 1. - available[r], r))
                balanced = sorted(deficient, key=lambda r: (1. - available[r], r))
                priority_order_changes += int(ordered != balanced)
        if policy == 'pirc_nosharing' or share_mode == 'none':
            remaining = 1.
            edge_used = {tuple(sorted((r, d))): 0. for r in range(G) for d in NEIGHBORS[r]}
        else:
            priority_gain = cfg['transfer_priority_gain'] if policy != 'traditional_share' else 0.
            method = 'pirc_share' if priority_gain else 'safe_fixed'
            params = {'risk_gain': priority_gain, 'slow_gain': 0., 'protect': 0.}
            moved, remaining, edge_used = transfer(
                available, req[t], inp['activity'][t], forecast[t],
                np.zeros(G), method, params, safe=np.zeros(G) if share_mode == 'balance' else safe,
                recipient_priority=local_priority)
            transferred += moved
            for phase in transfer_by_phase:
                if inp[phase][t]: transfer_by_phase[phase] += moved
        call = req[t] & (available >= 1. - 1e-8)
        served[t] = call
        failed = req[t] & ~call
        donor_surplus = np.maximum(0., available - q)
        for r in np.flatnonzero(failed):
            other = [d for d in range(G) if d != r]
            if np.any(donor_surplus[other] > 1e-8):
                global_crsr += 1
            need = 1. - available[r]
            possible = sum(max(0., min(donor_surplus[d],
                1. - edge_used[tuple(sorted((r, int(d))))]))
                for d in NEIGHBORS[r])
            if possible + 1e-8 >= need and remaining + 1e-8 >= need:
                adjacent_crsr += 1
        reserve = available - call.astype(float)
        if t % 512 == 383: preoutage_reserve.append(float(reserve.mean()))
        if policy != 'traditional_share' and cfg['lookahead_gain'] > 0.:
            logits = cfg['fast_gain'] * q + cfg['alloc_forecast_gain'] * forecast[t]
            if policy != 'pirc_noreserve':
                queued_supply = supply_ema * np.sum(queue, axis=0)
                projected = np.clip(reserve + queued_supply - DELAY * forecast[t], 0., CAP)
                logits += cfg['lookahead_gain'] * (CAP - projected) / CAP
        else:
            logits = 4. * q
            if policy != 'pirc_noreserve':
                logits += (2. if policy == 'traditional_share' else cfg['reserve_alloc_gain']) * np.maximum(0., (CAP - reserve) / CAP)
            if policy != 'traditional_share':
                logits += cfg['alloc_forecast_gain'] * forecast[t]
        # Causal precharge: allocate observed surplus to regions below capacity.
        burst_signal = np.clip((supply_ema - cfg['burst_threshold']) / 1., 0., 1.)
        logits += cfg['burst_alloc_gain'] * burst_signal * (CAP - reserve) / CAP
        queued_supply = supply_ema * np.sum(queue, axis=0)
        projected = np.clip(reserve + queued_supply - DELAY * forecast[t], 0., CAP)
        logits += cfg['burst_projected_gain'] * burst_signal * (CAP - projected) / CAP
        if allocation_mode == 'fixed':
            queue.append(np.full(G, 1. / G))
        elif allocation_mode == 'instant':
            queue.append(softmax(4. * inp['activity'][t]))
        elif allocation_mode == 'ema':
            queue.append(softmax(4. * slow_activity))
        else:
            queue.append(softmax(logits))
        assert np.all(reserve >= -1e-8) and np.all(reserve <= CAP + 1e-8)
    requests = int(req.sum())
    calls = int(served.sum())
    assert calls + int((req & ~served).sum()) == requests
    assert abs(G * INITIAL + supply.sum() - calls - spill - reserve.sum()) < 1e-4
    out = inp['outage']
    out_req = int(req[out].sum())
    base, extra = dense_mac_per_t()
    mask_order = served.reshape(inp['nclips'], T, G)
    mask = np.empty_like(mask_order)
    mask[inp['order']] = mask_order
    stats = {
        'requests': requests,
        'calls': calls,
        'high_calls_per_clip': calls / inp['nclips'],
        'high_call_rate': calls / (inp['nclips'] * T * G),
        'rcir': 1. - calls / requests,
        'outage_requests': out_req,
        'outage_calls': int(served[out].sum()),
        'outage_service': float(served[out].sum() / out_req),
        'global_crsr': global_crsr / requests,
        'adjacent_crsr': adjacent_crsr / requests,
        'overflow_rate': float(spill / (G * INITIAL + supply.sum())),
        'transferred_tokens': transferred,
        'competing_recipients': competing_recipients,
        'priority_order_changes': priority_order_changes,
        'safe_mean_per_region': safe_total / (n * G),
        'preoutage_reserve_per_region': float(np.mean(preoutage_reserve)) if preoutage_reserve else '',
        'normal_service': float(served[inp['normal']].sum() / req[inp['normal']].sum()),
        'burst_service': float(served[inp['burst']].sum() / req[inp['burst']].sum()),
        'transfer_normal': transfer_by_phase['normal'],
        'transfer_burst': transfer_by_phase['burst'],
        'transfer_outage': transfer_by_phase['outage'],
        'executed_mac_per_clip': T * base + calls / inp['nclips'] * extra + (T * G * 38 if utility_priority is not None else 0),
        'utility_predictor_mac_per_clip': T * G * 38 if utility_priority is not None else 0,
        'total_supply': float(supply.sum()),
        'original_requests_hash': hashlib.sha256(req.tobytes()).hexdigest(),
        'supply_hash': hashlib.sha256(supply.tobytes()).hexdigest(),
    }
    return mask, stats


def summarize(rows):
    ans = []
    for policy in ('fixed_low',) + POLICIES:
        sub = [r for r in rows if r['policy'] == policy]
        out = {'policy': policy, 'split': 'heldout_train_subject_validation',
               'scenario': SCENARIO, 'capacity': CAP,
               'n_clips': 252, 'n_supply_seeds': len(SEEDS)}
        for key in ('accuracy', 'high_calls_per_clip', 'high_call_rate', 'rcir',
                    'outage_service', 'executed_mac_per_clip', 'global_crsr',
                    'adjacent_crsr', 'overflow_rate', 'transferred_tokens'):
            vals = [float(r[key]) for r in sub if r[key] != '']
            out[key] = float(np.mean(vals)) if vals else ''
        ans.append(out)
    return ans


def save_csv(path, rows):
    assert rows
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'raw.csv').exists(), 'Refuse overwrite'
    fit = np.load('results/longrun_seed42/fit_rates.npy')
    cache = np.load('results/local_seed42/controllers/validation_cache.npz')
    assert fit.shape == (924, T, G)
    assert cache['rate'].shape == (252, T, G)
    ck = Path('results/local_seed42/best.pt')
    ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    assert ck_sha == '240d3305ec7876999d660af9051479cdcb4e51e8f6f943d4c712e3100aed811d'
    full_coef = np.asarray(json.loads(
        Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    assert full_coef.shape == (7,)
    cur_coef = current_only_coef(fit)
    rows = []
    base, _ = dense_mac_per_t()
    low_accuracy = float(np.mean(cache['low'].argmax(1) == cache['labels']))
    masks = {}
    for seed in SEEDS:
        inp = stream_inputs(cache['rate'], SCENARIO, seed)
        forecast_full = predict(inp, full_coef)
        forecast_cur = current_only_predict(inp, cur_coef)
        req_hash = hashlib.sha256(inp['requested'].tobytes()).hexdigest()
        supply_hash = hashlib.sha256(inp['supply'].tobytes()).hexdigest()
        rows.append({'policy': 'fixed_low', 'seed': seed,
                     'accuracy': low_accuracy, 'high_calls_per_clip': 0.,
                     'high_call_rate': 0., 'rcir': '', 'outage_service': '',
                     'executed_mac_per_clip': T * base,
                     'global_crsr': '', 'adjacent_crsr': '',
                     'overflow_rate': '', 'transferred_tokens': 0.,
                     'requests': 0, 'calls': 0, 'outage_requests': 0,
                     'outage_calls': 0, 'total_supply': inp['supply'].sum(),
                     'original_requests_hash': req_hash, 'supply_hash': supply_hash})
        for policy in POLICIES:
            forecast = forecast_cur if policy == 'pirc_nohistory' else forecast_full
            mask, stat = simulate(inp, forecast, policy)
            assert stat['requests'] == 32 * len(cache['labels'])
            assert stat['original_requests_hash'] == req_hash
            assert stat['supply_hash'] == supply_hash
            rows.append({'policy': policy, 'seed': seed,
                         'accuracy': accuracy(cache, mask), **stat})
            if seed == SEEDS[0]:
                masks[policy] = mask
        print('SEED', seed, 'finished', flush=True)
    save_csv(ROOT / 'raw.csv', rows)
    save_csv(ROOT / 'main_results.csv', summarize(rows))
    np.savez_compressed(ROOT / 'first_seed_masks.npz', **masks)
    (ROOT / 'manifest.json').write_text(json.dumps({
        'protocol_commit': PROTOCOL,
        'checkpoint_path': str(ck), 'checkpoint_sha256': ck_sha,
        'dataset': 'DVS128 Gesture', 'split': 'heldout training subjects',
        'n_validation_clips': 252, 'official_test_labels_used': False,
        'scenario': SCENARIO, 'capacity': CAP, 'initial': INITIAL,
        'allocation_delay': DELAY, 'seeds': SEEDS,
        'original_requests_per_clip': 32,
        'full_predictor_coef': full_coef.tolist(),
        'current_only_predictor_coef': cur_coef.tolist(),
        'result_rows': len(rows),
    }, indent=2))
    print('SAVED', ROOT, flush=True)


if __name__ == '__main__':
    main()
