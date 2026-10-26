"""Original six frozen baselines versus a PIRC-only compact High path."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from .compact_high import CompactHigh, CompactLocalSNN, compact_extra_mac
from .final_eval import actual_sparse
from .model import T, dense_mac_per_t
from .outage_sweep import stream_inputs
from .predictive_online import predict
from .seven_baselines import CKPATH, METHODS, SEEDS, load_cache, run_policy

ROOT = Path('results/pirc_only_compact_high')
FEATURE_ROOT = Path('results/compact_high_candidate_v2')
BASE_ROOT = Path('results/seven_baseline_head_v3')


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save(path,rows):
    assert rows and not path.exists()
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def physical_model(device,ck):
    model=CompactLocalSNN().to(device).eval()
    original=torch.load(CKPATH,map_location=device,weights_only=False)['model']
    missing,unexpected=model.load_state_dict(original,strict=False)
    assert set(missing)==set(model.compact.state_dict(prefix='compact.')) and not unexpected
    model.compact.load_state_dict(ck['compact'])
    return model


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'official_test_mixed_model_table.csv').exists(), 'Refuse overwrite'
    original_sha=hashlib.sha256(CKPATH.read_bytes()).hexdigest()
    compact_path=FEATURE_ROOT/'best.pt'
    compact_sha=hashlib.sha256(compact_path.read_bytes()).hexdigest()
    device=os.environ.get('PIRC_DEVICE', 'cuda:0')
    torch.set_num_threads(4)
    ck=torch.load(compact_path,map_location=device,weights_only=False)
    head=CompactHigh().to(device).eval()
    head.load_state_dict(ck['compact'])
    full=physical_model(device,ck)
    cfg=json.loads(Path('configs/resource_controller.json').read_text())['config']
    coef=np.asarray(json.loads(Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    base,_=dense_mac_per_t()
    audit=[]
    baseline_sha={}
    for split in ('validation','official_test'):
        original_raw=read_csv(BASE_ROOT/f'{split}_raw.csv')
        original_summary=read_csv(BASE_ROOT/f'{split}_summary.csv')
        baseline_sha[split]=hashlib.sha256((BASE_ROOT/f'{split}_summary.csv').read_bytes()).hexdigest()
        old_pirc={int(r['seed']):r for r in original_raw if r['method']=='pirc'}
        assert len(old_pirc)==len(SEEDS[split])
        cache=load_cache(split)
        f=np.load(FEATURE_ROOT/f'{split}_features.npz')
        assert np.array_equal(f['labels'],cache['labels'])
        assert np.max(np.abs(f['low']-cache['low']))<2e-5
        with torch.no_grad():
            x=torch.as_tensor(f['features'],device=device)
            corr=head(x.reshape(-1,x.shape[-1])).reshape(len(x),T,4,11).cpu().numpy()
        pirc_rows=[]
        for seed in SEEDS[split]:
            inp=stream_inputs(cache['rate'],'d64_v50',seed)
            forecast=predict(inp,coef)
            mask,st=run_policy(inp,forecast,'pirc',cfg)
            old=old_pirc[seed]
            assert st['original_requests_hash']==old['original_requests_hash']
            assert st['supply_hash']==old['supply_hash']
            for key in ('rcir','outage_service','high_calls_per_clip'):
                assert abs(st[key]-float(old[key]))<1e-12
            logits=cache['low']+np.einsum('ntg,ntgc->nc',mask,corr,optimize=True)/T
            pirc_rows.append({'split':split,'seed':seed,'method':'pirc_compact_high',
                'accuracy':float(np.mean(logits.argmax(1)==cache['labels'])),
                'rcir':st['rcir'],'outage_service':st['outage_service'],
                'executed_mac_per_clip':T*base+st['high_calls_per_clip']*compact_extra_mac(),
                'high_calls_per_clip':st['high_calls_per_clip'],
                'original_requests_hash':st['original_requests_hash'],'supply_hash':st['supply_hash']})
            if seed==SEEDS[split][0]:
                frames=np.load(f'data/prepared/{"train" if split=="validation" else "test"}_frames.npy',mmap_mode='r')
                ids=f['indices'][:16]
                physical,ms=actual_sparse(full,frames[ids],mask[:16],device,return_logits=True)
                diff=float(np.max(np.abs(physical-logits[:16])))
                assert diff<.003 and np.array_equal(physical.argmax(1),logits[:16].argmax(1))
                audit.append({'split':split,'nclips':16,'max_logit_diff':diff,
                              'classification_mismatches':0,'first_pass_gpu_network_ms_per_clip':ms})
            print('PIRC_DONE',split,seed,flush=True)
        save(ROOT/f'{split}_pirc_raw.csv',pirc_rows)
        rows=[]
        for method in METHODS:
            if method=='pirc':continue
            r=next(r for r in original_summary if r['method']==method)
            rows.append({'method':method,'split':split,'accuracy':float(r['accuracy']),
                'rcir':float(r['rcir']),'outage_service':float(r['outage_service']),
                'executed_mac_per_clip':float(r['executed_mac_per_clip']),
                'checkpoint':'original_high_head','checkpoint_sha256':original_sha,
                'baseline_result_unchanged':True})
        rows.append({'method':'pirc_compact_high','split':split,
            **{k:float(np.mean([r[k] for r in pirc_rows])) for k in
               ('accuracy','rcir','outage_service','executed_mac_per_clip')},
            'checkpoint':'compact_high_head','checkpoint_sha256':compact_sha,
            'baseline_result_unchanged':False})
        assert len(rows)==7
        save(ROOT/f'{split}_mixed_model_table.csv',rows)
        print('TABLE',split,json.dumps(rows),flush=True)
    save(ROOT/'physical_audit.csv',audit)
    (ROOT/'manifest.json').write_text(json.dumps({'comparison':'mixed-checkpoint end-to-end system, not isolated scheduling',
        'original_baseline_checkpoint':str(CKPATH),'original_baseline_sha256':original_sha,
        'original_baseline_summary_sha256':baseline_sha,
        'pirc_compact_checkpoint':str(compact_path),'pirc_compact_sha256':compact_sha,
        'pirc_raw_requests_supply_and_served_masks_match_original':True,
        'dataset':'DVS128 Gesture','scenario':'d64_v50','supply_seeds':SEEDS,
        'official_test_labels_used_for_tuning':False},indent=2))

if __name__=='__main__':main()
