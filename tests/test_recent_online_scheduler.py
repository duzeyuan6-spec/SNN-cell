import numpy as np
from local_pirc.outage_sweep import stream_inputs
from local_pirc.recent_online_scheduler import external_simulate


def test_two_adapters_conserve_resources_and_are_causal():
    rates = np.random.default_rng(42).random((32, 16, 4))
    inp = stream_inputs(rates, 'd64_v50', 1201)
    forecast = np.full(inp['requested'].shape, 0.5)
    for method, param in [('dsm_2025_adapted', 0.5), ('ecss_2025_adapted', 0.5)]:
        mask, stats, trace = external_simulate(inp, forecast, method, param, return_trace=True)
        assert mask.shape == (32, 16, 4)
        assert 0 <= stats['rcir'] <= 1 and 0 <= stats['outage_service'] <= 1
        assert stats['requests'] == 32 * 32
        changed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
        changed['supply'][300:] = 0.0
        changed['activity'][300:] = 0.0
        other_forecast = forecast.copy(); other_forecast[300:] = 0.0
        # Compare state up to t=299 only: later inputs must not change earlier actions.
        _, _, other_trace = external_simulate(changed, other_forecast, method, param, return_trace=True)
        for t in range(300):
            assert np.allclose(trace[t][0], other_trace[t][0])
            assert np.allclose(trace[t][1], other_trace[t][1])
            assert trace[t][2] == other_trace[t][2]
