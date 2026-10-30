import sys
import json
import numpy as np
import torch
from local_pirc.pirc_only_compact_eval import physical_model
from local_pirc.compact_high import CompactHigh
from local_pirc.recent_online_eval import load_split, CHECKPOINT
from local_pirc.recent_online_scheduler import external_simulate
from local_pirc.outage_sweep import stream_inputs
from local_pirc.predictive_online import predict
from local_pirc.final_eval import actual_sparse

device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
coef = np.asarray(json.loads(open('results/predictive_online_seed42/predictor.json').read())['coef'])
ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = physical_model(device, ck)
head = CompactHigh().eval(); head.load_state_dict(ck["compact"])
rate, low, labels, corr = load_split('official_test', head)
inp = stream_inputs(rate, 'd64_v50', 1301)
forecast = predict(inp, coef)
f = np.load('results/compact_high_candidate_v2/official_test_features.npz')
frames = np.load('data/prepared/test_frames.npy', mmap_mode='r')
for method, param in [('dsm_2025_adapted', 0.0), ('ecss_2025_adapted', 0.5)]:
    mask, st, _ = external_simulate(inp, forecast, method, param)
    cache_logits = low + np.einsum('ntg,ntgc->nc', mask, corr, optimize=True) / 16
    physical, ms = actual_sparse(model, frames[f['indices'][:16]], mask[:16], device, return_logits=True)
    diff = float(np.max(np.abs(physical - cache_logits[:16])))
    mismatch = int(np.sum(physical.argmax(1) != cache_logits[:16].argmax(1)))
    print(method, 'max_logit_diff', diff, 'mismatch', mismatch, 'first_pass_gpu_ms_per_clip', ms, flush=True)
    assert diff < 0.003 and mismatch == 0
