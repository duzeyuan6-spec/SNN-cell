"""Causal online allocation under a shared supply trace and atomic High calls."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from .model import T, GROUPS, dense_mac_per_t
from .outage_sweep import DELAY, SCENARIOS, stream_inputs

G = GROUPS
FIT_SEEDS = (701, 702, 703)
VAL_SEEDS = (711, 712, 713)
FIT_POLICY_CLIPS = 128
VAL_CLIPS = 64
CAPS = (4., 12.)
H = DELAY


def features(activity, requested):
    """All features at t use data through t; reset only neural history per clip."""
    n = len(activity)
    x = np.zeros((n, G, 7), dtype=np.float64)
    fast_a = np.zeros(G); slow_a = np.zeros(G)
    fast_q = np.zeros(G); slow_q = np.zeros(G)
    for t in range(n):
        if t % T == 0:
            fast_a[:] = 0.; slow_a[:] = 0.
            fast_q[:] = 0.; slow_q[:] = 0.
        a = activity[t]; q = requested[t].astype(float)
        fast_a = .75 * fast_a + .25 * a
        slow_a = .9375 * slow_a + .0625 * a
        fast_q = .75 * fast_q + .25 * q
        slow_q = .9375 * slow_q + .0625 * q
        x[t, :, 0] = 1.
        x[t, :, 1] = a
        x[t, :, 2] = fast_a
        x[t, :, 3] = slow_a
        x[t, :, 4] = q
        x[t, :, 5] = fast_q
        x[t, :, 6] = slow_q
    return x


def target(requested):
    """Training target, never computed from validation future for policy decisions."""
    y = np.zeros(requested.shape, dtype=np.float64)
    for h in range(1, H + 1):
        y[:-h] += requested[h:] / H
    return y


def fit_predictor(rates):
    gram = np.zeros((7, 7)); rhs = np.zeros(7)
    for seed in FIT_SEEDS:
        inp = stream_inputs(rates, 'd64_v50', seed)
        x = features(inp['activity'], inp['requested']).reshape(-1, 7)
        y = target(inp['requested']).reshape(-1)
        gram += x.T @ x; rhs += x.T @ y
    reg = 1e-3 * np.eye(7); reg[0, 0] = 0.
    coef = np.linalg.solve(gram + reg, rhs)
    return coef


def predict(inp, coef):
    x = features(inp['activity'], inp['requested'])
    return np.clip(x @ coef, 0., 1.)


def softmax(x):
    z = np.exp(x - np.max(x))
    return z / z.sum()


def simulate_online(inp, pred, method, params, cap):
    """Supply allocation is chosen after step t; reaches groups at t+4."""
    assert method in ('threshold', 'predictive_threshold', 'predictive_pirc')
    n = len(inp['supply']); supply = inp['supply']; requested = inp['requested']
    assert pred.shape == requested.shape
    reserve = np.full(G, 2., dtype=np.float64)
    queue = [np.full(G, 1. / G) for _ in range(DELAY)]
    served = np.zeros_like(requested)
    spill_total = 0.; supply_ema = 1.6; deferred_total = 0
    slow_error = np.zeros(G); debt = np.zeros(G)
    for t in range(n):
        weights = queue.pop(0)
        incoming = supply[t] * weights
        pre = reserve + incoming
        spill_total += np.maximum(pre - cap, 0.).sum()
        available = np.minimum(pre, cap)
        eligible = requested[t] & (available >= 1. - 1e-9)
        call = eligible.copy()
        if method != 'threshold' and params.get('hold_level') is not None:
            supply_seen = .9 * supply_ema + .1 * supply[t]
            queued_supply = supply_seen * np.sum(queue, axis=0)
            future_shortage = (H * pred[t] - (available - eligible.astype(float)) - queued_supply) / cap
            lower_priority = inp['activity'][t] < np.max(inp['activity'][t, requested[t]])
            call &= ~(lower_priority & (future_shortage > params['hold_level']))
        deferred_total += int((eligible & ~call).sum())
        served[t] = call
        missed = requested[t] & ~call
        reserve = available - call.astype(float)
        supply_ema = .9 * supply_ema + .1 * supply[t]
        expected_incoming = supply_ema * np.sum(queue, axis=0)
        forecast_shortage = np.maximum(0., H * pred[t] - reserve - expected_incoming) / cap
        base = params['gain'] * requested[t].astype(float)
        base += params['boost'] * np.maximum(0., (cap - reserve) / cap)
        if method == 'threshold':
            logits = base
        elif method == 'predictive_threshold':
            logits = base + params['forecast_gain'] * forecast_shortage
        else:
            net_error = np.clip((H * pred[t] - reserve - expected_incoming) / cap, -1., 1.)
            rho = params['rho']
            slow_error = rho * slow_error + (1. - rho) * net_error
            debt = .9 * debt + .1 * missed.astype(float)
            logits = base + params['forecast_gain'] * forecast_shortage
            logits += params['slow_gain'] * slow_error + debt
        queue.append(softmax(logits))
    calls = int(served.sum()); requests = int(requested.sum())
    outage = inp['outage']
    outage_calls = int(served[outage].sum())
    outage_requests = int(requested[outage].sum())
    total_supply = 2. * G + float(supply.sum())
    assert calls + int((requested & ~served).sum()) == requests
    assert abs(total_supply - calls - spill_total - reserve.sum()) < 1e-4
    base_mac, extra_mac = dense_mac_per_t()
    extra_controller = G * (0 if method == 'threshold' else 12 if method == 'predictive_threshold' else 20)
    stats = {'requests': requests, 'calls': calls, 'interrupted': requests - calls,
             'rcir': (requests - calls) / requests, 'deferred_calls': deferred_total,
             'outage_requests': outage_requests, 'outage_calls': outage_calls,
             'outage_service': outage_calls / outage_requests,
             'spill_tokens': spill_total, 'spill_fraction': spill_total / total_supply,
             'supply_tokens': float(supply.sum()), 'terminal_reserve': float(reserve.sum()),
             'mac_per_clip': T * base_mac + calls / inp['nclips'] * extra_mac + T * extra_controller,
             'high_mac_per_clip': calls / inp['nclips'] * extra_mac,
             'high_calls_per_clip': calls / inp['nclips']}
    mask = served.reshape(inp['nclips'], T, G)
    original = np.empty_like(mask); original[inp['order']] = mask
    return original, stats, served


def objective(stats):
    return .5 * stats['rcir'] + .5 * (1. - stats['outage_service']) + .1 * stats['spill_fraction']


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def accuracy(cache, mask):
    n = len(mask); low = cache['low'][:n]; corr = cache['corr'][:n]
    logits = low + np.einsum('ntg,ntgc->nc', mask, corr, optimize=True) / T
    y = cache['labels'][:n]
    return float(np.mean(np.argmax(logits, axis=1) == y))
