"""Causal, resource-conserving adapters of two 2025 online edge schedulers.

These are SNN-specific adaptations, not reimplementations of the papers' native
DVFS, service migration, or serverless execution environments. All evaluated methods
use identical exogenous supply, High requests, capacity and adjacent transfer limits.
"""
import hashlib
import numpy as np
from .local_sharing import G, NEIGHBORS, transfer
from .model import T
from .outage_sweep import DELAY

CAP = 4.0
INITIAL = 2.0


def softmax(logits):
    x = np.exp(logits - np.max(logits))
    return x / x.sum()


def external_simulate(inp, forecast, method, parameter, *, return_trace=False):
    """DSM: forecast supply-demand regime and move capacity from surplus to deficit.

    ECSS: select sufficient/constrained/inadequate scheduler from mean residual
    resource and observed demand. At each t only current/past signals are read.
    """
    assert method in ('dsm_2025_adapted', 'ecss_2025_adapted')
    supply = inp['supply']; req = inp['requested']; activity = inp['activity']
    n = len(supply)
    assert req.shape == forecast.shape == (n, G)
    reserve = np.full(G, INITIAL, dtype=np.float64)
    queue = [np.full(G, 1.0 / G) for _ in range(DELAY)]
    served = np.zeros_like(req)
    supply_ema = 1.6
    spill = 0.0; moved_total = 0.0
    trace = []
    for t in range(n):
        incoming = supply[t] * queue.pop(0)
        pre = reserve + incoming
        spill += np.maximum(pre - CAP, 0.0).sum()
        available = np.minimum(pre, CAP)
        q = req[t].astype(float)
        supply_ema = 0.9 * supply_ema + 0.1 * supply[t]
        pending = supply_ema * np.sum(queue, axis=0)
        if method == 'dsm_2025_adapted':
            # Paper's predicted energy supply/demand groups become local resource
            # surplus/deficit groups; indivisible High calls replace DVFS.
            projected = available + pending - DELAY * forecast[t]
            deficit = np.maximum(-projected, 0.0)
            safe = np.clip(float(parameter) - projected, 0.0, CAP)
            recipient = deficit + 2.0 * q + (1.0 - available / CAP)
            logits_extra = 2.0 * deficit + 1.5 * (CAP - available) / CAP
        else:
            # Paper's average-energy pre-scheduler selects a regime. External
            # helper nodes are unavailable; adjacent token transfer is allowed.
            margin = float(available.mean() - q.mean())
            if margin >= float(parameter):  # sufficient
                safe = np.zeros(G)
                logits_extra = (CAP - available) / CAP
            elif margin >= 0.0:  # constrained but functional
                safe = np.full(G, 0.5 * float(parameter))
                logits_extra = 2.0 * (CAP - available) / CAP
            else:  # inadequate
                safe = np.zeros(G)
                logits_extra = 3.0 * (CAP - available) / CAP
            recipient = q + 0.25 * (CAP - available) / CAP
        moved, _, _ = transfer(available, req[t], activity[t], forecast[t],
                                np.zeros(G), 'safe_fixed',
                                {'risk_gain': 0.0, 'slow_gain': 0.0, 'protect': 0.0},
                                safe=safe, recipient_priority=recipient)
        moved_total += moved
        call = req[t] & (available >= 1.0 - 1e-8)
        served[t] = call
        reserve = available - call.astype(float)
        queue.append(softmax(4.0 * q + logits_extra))
        if return_trace:
            trace.append((reserve.copy(), incoming.copy(), float(moved)))
        assert np.all(reserve >= -1e-8) and np.all(reserve <= CAP + 1e-8)
    requests = int(req.sum()); calls = int(served.sum())
    assert requests == 32 * inp['nclips']
    assert abs(G * INITIAL + supply.sum() - calls - spill - reserve.sum()) < 1e-4
    outage_requests = int(req[inp['outage']].sum())
    mask_order = served.reshape(inp['nclips'], T, G)
    mask = np.empty_like(mask_order)
    mask[inp['order']] = mask_order
    stats = dict(requests=requests, calls=calls, rcir=1.0 - calls / requests,
                 outage_requests=outage_requests,
                 outage_calls=int(served[inp['outage']].sum()),
                 outage_service=float(served[inp['outage']].sum() / outage_requests),
                 high_calls_per_clip=calls / inp['nclips'],
                 spill_tokens=float(spill), transferred_tokens=float(moved_total),
                 total_supply=float(supply.sum()),
                 original_requests_hash=hashlib.sha256(req.tobytes()).hexdigest(),
                 supply_hash=hashlib.sha256(supply.tobytes()).hexdigest())
    return mask, stats, trace
