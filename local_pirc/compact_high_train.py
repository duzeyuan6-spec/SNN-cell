"""Train one compact High candidate with frozen Backbone and mixed deployable masks."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .compact_high import CompactHigh
from .outage_sweep import stream_inputs
from .predictive_online import predict
from .seven_baselines import run_policy

ROOT = Path('results/compact_high_candidate_v2')
METHODS = ('fixed_allocation','balanced_sharing','predictive_reserve','pirc')

def banks(split, cache):
    seeds = (701,702) if split == 'fit' else (1201,1202)
    cfg = json.loads(Path('configs/resource_controller.json').read_text())['config']
    coef = np.asarray(json.loads(Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    masks = []
    for seed in seeds:
        inp = stream_inputs(cache['rate'], 'd64_v50', seed)
        forecast = predict(inp, coef)
        for method in METHODS:
            mask, stats = run_policy(inp, forecast, method, cfg)
            masks.append(mask)
    return np.stack(masks)

def eval_head(net, feat, low, labels, masks):
    net.eval()
    with torch.no_grad():
        corr = net(feat.reshape(-1,feat.shape[-1])).reshape(*feat.shape[:3],11)
        out = []
        for m in masks:
            logits = low + (corr * m[...,None]).sum((1,2)) / 16
            out.append((float((logits.argmax(1)==labels).float().mean()),float(F.cross_entropy(logits,labels))))
    return out

def main():
    assert not (ROOT / 'best.pt').exists(), 'Refuse overwrite'
    torch.manual_seed(42);np.random.seed(42)
    torch.set_num_threads(4)
    device=os.environ.get('PIRC_DEVICE', 'cuda:0')
    fit = np.load(ROOT / 'fit_features.npz')
    val = np.load(ROOT / 'validation_features.npz')
    fit_bank = banks('fit',fit)
    val_bank = banks('validation',val)
    tf = torch.tensor(fit['features'], device=device)
    vf = torch.tensor(val['features'], device=device)
    tl = torch.tensor(fit['low'], device=device)
    vl = torch.tensor(val['low'], device=device)
    ty = torch.tensor(fit['labels'], device=device, dtype=torch.long)
    vy = torch.tensor(val['labels'], device=device, dtype=torch.long)
    tm = torch.tensor(fit_bank, device=device,dtype=torch.float32)
    vm = torch.tensor(val_bank, device=device,dtype=torch.float32)
    net = CompactHigh().to(device)
    opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    rng=np.random.default_rng(42)
    history=[];best=float('inf');bad=0
    for epoch in range(1,101):
        net.train()
        perm=rng.permutation(len(tf))
        losses=[]
        for start in range(0,len(perm),64):
            ix=torch.tensor(perm[start:start+64],device=device)
            choices=torch.tensor(rng.integers(len(tm),size=len(ix)),device=device)
            m=tm[choices,ix]
            feat=tf[ix]
            corr=net(feat.reshape(-1,feat.shape[-1])).reshape(*feat.shape[:3],11)
            logits=tl[ix]+(corr*m[...,None]).sum((1,2))/16
            loss=F.cross_entropy(logits,ty[ix])
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            losses.append(float(loss.detach()))
        scores=eval_head(net,vf,vl,vy,vm)
        val_ce=float(np.mean([v[1] for v in scores]))
        row={'epoch':epoch,'train_ce':float(np.mean(losses)),'val_ce':val_ce,
             **{f'{METHODS[i%4]}_seed{i//4}_acc':v[0] for i,v in enumerate(scores)}}
        history.append(row)
        print(json.dumps(row),flush=True)
        if val_ce < best-1e-4:
            best=val_ce;bad=0
            torch.save({'compact':net.state_dict(),'epoch':epoch,'val_ce':val_ce,
                        'backbone_checkpoint':'results/refine_adapt_subject_seed42/best.pt',
                        'train_masks':METHODS,'fit_seeds':(701,702),'val_seeds':(1201,1202)},ROOT/'best.pt')
        else:
            bad+=1
            if bad>=15:break
    with (ROOT/'train_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    print('BEST',best,flush=True)

if __name__=='__main__':main()
