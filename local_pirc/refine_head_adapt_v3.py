"""Fit only the existing local correction head on feasible full-recording masks."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import T
from .outage_sweep import stream_inputs
from .predictive_online import predict
from .unified_system import simulate

ROOT = Path('results/refine_adapt_subject_seed42')
POLICIES = ('traditional_share','predictive_share','pirc_full')


def masks_for(rate, seeds, cfg, coef):
    banks = []
    meta = []
    for seed in seeds:
        inp = stream_inputs(rate, 'd64_v50', seed)
        forecast = predict(inp, coef)
        for policy in POLICIES:
            mask, st = simulate(inp, forecast, policy, cfg)
            assert st['requests'] == 32*len(rate)
            banks.append(mask)
            meta.append({'seed':seed,'policy':policy,
                         'requests_hash':st['original_requests_hash'],
                         'supply_hash':st['supply_hash']})
    return np.asarray(banks), meta


@torch.no_grad()
def evaluate(head, feat, low, labels, masks):
    head.eval()
    corr = head(feat)
    rows = []
    for m in masks:
        logits = low + torch.einsum('ntg,ntgc->nc', m, corr)/T
        rows.append((float(F.cross_entropy(logits,labels)),
                     float((logits.argmax(1)==labels).float().mean())))
    return float(np.mean([x[0] for x in rows])),float(np.mean([x[1] for x in rows]))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    assert not (ROOT/'best.pt').exists(), 'Refuse overwrite'
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    device=os.environ.get('PIRC_DEVICE', 'cuda:0')
    ck=torch.load('results/local_seed42/best.pt',map_location='cpu',weights_only=False)
    lock=json.loads(Path('configs/resource_controller.json').read_text())
    coef=np.asarray(json.loads(Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    data=np.load('results/unified_followup/fit_cache.npz')
    feat=np.load('results/refine_adapt_seed42/fit_features.npz')['features']
    assert len(feat)==len(data['rate'])==924
    import re
    paths=json.loads(Path('data/prepared/train_sources.json').read_text())
    subjects=np.asarray([re.search(r'user(\d+)',paths[int(i)]).group(1)
                         for i in data['indices']])
    unique=np.array(sorted(set(subjects)))
    split_rng=np.random.default_rng(2026)
    heldout=sorted(split_rng.permutation(unique)[:4].tolist())
    sel=np.flatnonzero(np.isin(subjects,heldout))
    tr=np.flatnonzero(~np.isin(subjects,heldout))
    labels=data['labels']
    assert np.all(np.bincount(labels[sel],minlength=11)>0)
    assert np.all(np.bincount(labels[tr],minlength=11)>0)
    print('HELDOUT_SUBJECTS',heldout,'SPLIT_COUNTS',
          np.bincount(labels[tr],minlength=11).tolist(),
          np.bincount(labels[sel],minlength=11).tolist(),flush=True)
    train_masks,train_meta=masks_for(data['rate'][tr],range(711,717),lock['config'],coef)
    select_masks,select_meta=masks_for(data['rate'][sel],range(717,720),lock['config'],coef)
    def tensor(x,dtype=None):return torch.as_tensor(x,dtype=dtype,device=device)
    x=tensor(feat[tr]);l=tensor(data['low'][tr]);y=tensor(data['labels'][tr],torch.long)
    xsel=tensor(feat[sel]);lsel=tensor(data['low'][sel]);ysel=tensor(data['labels'][sel],torch.long)
    mask_tr=tensor(train_masks,torch.float32);mask_sel=tensor(select_masks,torch.float32)
    head=nn.Linear(72,11).to(device)
    head.weight.data.copy_(ck['model']['refine.head.weight'].to(device))
    head.bias.data.copy_(ck['model']['refine.head.bias'].to(device))
    opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    base_ce,base_acc=evaluate(head,xsel,lsel,ysel,mask_sel)
    print('ORIGINAL_SELECTION',base_ce,base_acc,flush=True)
    best=base_ce;stale=0
    logpath=ROOT/'training.jsonl'
    assert not logpath.exists(), 'Refuse overwrite'
    rng=np.random.default_rng(42)
    with logpath.open('w') as f:
        for epoch in range(1,41):
            head.train()
            order=rng.permutation(len(x))
            losses=[]
            for start in range(0,len(order),64):
                ids=order[start:start+64]
                bank_idx=rng.integers(0,len(train_masks),size=len(ids))
                m=mask_tr[bank_idx,ids]
                corr=head(x[ids])
                logits=l[ids]+torch.einsum('ntg,ntgc->nc',m,corr)/T
                loss=F.cross_entropy(logits,y[ids])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            ce,acc=evaluate(head,xsel,lsel,ysel,mask_sel)
            row={'epoch':epoch,'train_ce':float(np.mean(losses)),
                 'select_ce':ce,'select_accuracy':acc}
            f.write(json.dumps(row)+'\n');f.flush()
            print('EPOCH',json.dumps(row),flush=True)
            if ce<best-1e-4:
                best=ce;stale=0
                state=ck['model'].copy()
                state['refine.head.weight']=head.weight.detach().cpu().clone()
                state['refine.head.bias']=head.bias.detach().cpu().clone()
                torch.save({'model':state,'source_checkpoint':'results/local_seed42/best.pt',
                            'epoch':epoch,'select_ce':ce,'select_accuracy':acc,
                            'train_indices':data['indices'][tr],
                            'select_indices':data['indices'][sel],
                            'only_head_updated':True},ROOT/'best.pt')
            else:
                stale+=1
                if stale>=6:break
    (ROOT/'config.json').write_text(json.dumps({
        'training_protocol':'REFINE_ADAPT_PROTOCOL_V3.md',
        'controller_lock':'configs/resource_controller.json',
        'train_n':len(x),'select_n':len(xsel),
        'train_positions':tr.tolist(),'select_positions':sel.tolist(),
        'heldout_subjects':heldout,
        'train_seeds':list(range(711,717)),'select_seeds':list(range(717,720)),
        'policies':POLICIES,'original_selection_ce':base_ce,
        'original_selection_accuracy':base_acc,'best_selection_ce':best,
        'train_mask_meta':train_meta,'select_mask_meta':select_meta},indent=2))
    print('BEST',best,'CHECKPOINT',ROOT/'best.pt',flush=True)


if __name__=='__main__':main()
