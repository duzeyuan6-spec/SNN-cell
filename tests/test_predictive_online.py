import numpy as np
from local_pirc.model import T, GROUPS
from local_pirc.outage_sweep import stream_inputs
from local_pirc.predictive_online import features, predict, simulate_online


def test_features_are_causal_and_reset_per_clip():
    activity = np.arange(2*T*GROUPS, dtype=float).reshape(2*T, GROUPS) / 100
    req = np.zeros_like(activity, dtype=bool); req[:, :2] = True
    before = features(activity, req)
    altered = activity.copy(); altered[T+3:] = 99.
    after = features(altered, req)
    np.testing.assert_array_equal(before[:T+3], after[:T+3])
    np.testing.assert_allclose(before[T, :, 2], .25*activity[T])


def test_online_resource_accounting_and_equal_exogenous_requests():
    rates = np.zeros((32, T, GROUPS), dtype=float)
    rates[:, :, 0] = .8; rates[:, :, 1] = .7
    inp = stream_inputs(rates, 'd64_v50', 17)
    forecast = np.full_like(inp['activity'], .5)
    params = {'gain':4., 'boost':2.}
    mask, stats, served = simulate_online(inp, forecast, 'threshold', params, 4.)
    assert mask.sum() == served.sum() == stats['calls']
    assert stats['requests'] == int(inp['requested'].sum())
    assert stats['calls'] + stats['interrupted'] == stats['requests']
    assert abs(2*GROUPS + inp['supply'].sum() - stats['calls']
               - stats['spill_tokens'] - stats['terminal_reserve']) < 1e-4
    p = dict(params, forecast_gain=2., rho=.9, slow_gain=2.)
    _, st, _ = simulate_online(inp, forecast, 'predictive_pirc', p, 4.)
    assert st['requests'] == stats['requests']
    assert st['supply_tokens'] == stats['supply_tokens']


def test_active_deferral_is_counted_as_interruption():
    rates = np.zeros((32, T, GROUPS), dtype=float)
    rates[:, :, 0] = .8; rates[:, :, 1] = .7
    inp = stream_inputs(rates, 'd64_v50', 18)
    pred = np.full_like(inp['activity'], .9)
    params = {'gain': 4., 'boost': 2., 'forecast_gain': 1., 'hold_level': -10.}
    _, st, _ = simulate_online(inp, pred, 'predictive_threshold', params, 4.)
    assert st['deferred_calls'] > 0
    assert st['calls'] + st['interrupted'] == st['requests']
    assert abs(2*GROUPS + inp['supply'].sum() - st['calls']
               - st['spill_tokens'] - st['terminal_reserve']) < 1e-4
