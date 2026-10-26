import numpy as np

from local_pirc.outage_sweep import stream_inputs
from local_pirc.seven_baselines import METHODS, run_policy


def test_all_seven_keep_original_requests_and_supply():
    rate = np.random.default_rng(8).random((32, 16, 4))
    inp = stream_inputs(rate, 'd64_v50', 75)
    forecast = np.full((32 * 16, 4), .4)
    cfg = {'safe_floor': .5, 'safe_gap_gain': .5, 'safe_forecast_gain': 1.,
           'burst_gain': 2., 'fast_gain': 0., 'lookahead_gain': 2.,
           'alloc_forecast_gain': 1.}
    hashes = set()
    for method in METHODS:
        mask, st = run_policy(inp, forecast, method, cfg)
        assert mask.shape == (32, 16, 4)
        assert st['requests'] == 32 * 32
        assert st['calls'] == int(mask.sum())
        assert st['total_supply'] == float(inp['supply'].sum())
        assert st['executed_mac_per_clip'] > 0
        hashes.add((st['original_requests_hash'], st['supply_hash']))
        if method in ('fixed_allocation', 'instant_gating', 'ema_gating'):
            assert st['transferred_tokens'] == 0
    assert len(hashes) == 1
