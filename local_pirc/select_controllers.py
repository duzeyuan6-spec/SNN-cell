"""Validation-only controller calibration and local utility diagnostics."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from .model import LocalSNN,T,GROUPS,dense_mac_per_t
from .controller import schedule


def load_model(checkpoint,device):
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    model=LocalSNN().to(device).eval();model.load_state_dict(ck['model'])
    for p in model.parameters():p.requires_grad_(False)
    return model,ck


@torch.no_grad()
def cache(model,frames,indices,device,batch=16):
    low_all=[];corr_all=[];rate_all=[]
    for start in range(0,len(indices),batch):
        ids=indices[start:start+batch]
        x=torch.as_tensor(np.array(frames[ids],copy=True),device=device)
        lows=[];corrs=[];rates=[]
        for t in range(T):
            low,local=model.step(x[:,t]);lows.append(low)
            rates.append(model.activity(local))
            patches=model.patches(local)
            region_ids=torch.arange(len(x)*GROUPS,device=device)%GROUPS
            corr=model.refine(patches,region_ids).reshape(len(x),GROUPS,11)
            corrs.append(corr)
        low_all.append(torch.stack(lows,1).mean(1).cpu().numpy())
        corr_all.append(torch.stack(corrs,1).cpu().numpy())
        rate_all.append(torch.stack(rates,1).cpu().numpy())
        model.reset()
    return np.concatenate(low_all),np.concatenate(corr_all),np.concatenate(rate_all)


def score(low,corr,labels,mask):
    logits=low+np.einsum('ntg,ntgc->nc',mask,corr,optimize=True)/T
    return {'accuracy':float((logits.argmax(1)==labels).mean()),
            'ce':float(F.cross_entropy(torch.from_numpy(logits),torch.as_tensor(labels)))}


def threshold_for_rate(rate,method,target,scenario,params):
    lo,hi=-1.,2.
    for _ in range(24):
        mid=(lo+hi)/2
        _,stat=schedule(rate,method,threshold=mid,scenario=scenario,**params)
        if stat['mean_calls']>target:lo=mid
        else:hi=mid
    options=[]
    for th in (lo,hi):
        mask,stat=schedule(rate,method,threshold=th,scenario=scenario,**params)
        options.append((abs(stat['mean_calls']-target),th,mask,stat))
    return min(options,key=lambda x:x[0])[1:]


def diagnose(low,corr,rate,labels):
    y=torch.as_tensor(labels)
    base_ce=F.cross_entropy(torch.from_numpy(low),y,reduction='none').numpy()
    corrected=low[:,None,None,:]+corr/T
    z=corrected.reshape(-1,11)
    yy=y[:,None,None].expand(len(y),T,GROUPS).reshape(-1)
    ce=F.cross_entropy(torch.from_numpy(z),yy,reduction='none').numpy().reshape(len(y),T,GROUPS)
    benefit=base_ce[:,None,None]-ce
    return {'single_call_positive_fraction':float((benefit>0).mean()),
      'mean_single_call_ce_benefit':float(benefit.mean()),
      'rate_benefit_spearman':float(spearmanr(rate.ravel(),benefit.ravel()).statistic),
      'rate_cv':float(rate.std()/max(rate.mean(),1e-9)),
      'region_mean_rates':rate.mean((0,1)).tolist(),
      'region_mean_benefit':benefit.mean((0,1)).tolist()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='results/local_seed42/best.pt')
    ap.add_argument('--data',default='data/prepared')
    ap.add_argument('--out',default='results/local_seed42/controllers')
    ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'locked.json').exists(), 'Already locked; do not retune.'
    torch.set_num_threads(4);model,ck=load_model(a.checkpoint,a.device)
    frames=np.load(Path(a.data)/'train_frames.npy',mmap_mode='r')
    labels=np.load(Path(a.data)/'train_labels.npy')
    val=np.asarray(ck['val_indices']);low,corr,rate=cache(model,frames,val,a.device)
    y=labels[val]
    np.savez_compressed(out/'validation_cache.npz',indices=val,low=low,corr=corr,rate=rate,labels=y)
    utility=diagnose(low,corr,rate,y)
    (out/'local_utility.json').write_text(json.dumps(utility,indent=2))
    scenario='bottleneck'
    fixed_mask,fixed_stat=schedule(rate,'fixed',scenario=scenario)
    target=fixed_stat['mean_calls']
    rows=[];chosen={}
    def add(method,params,th,mask,stat):
        m=score(low,corr,y,mask)
        row={'method':method,'params':json.dumps(params,sort_keys=True),
          'threshold':float(th),**m,
          'mean_calls':stat['mean_calls'],'mean_spill':stat['mean_spill'],
          'depletion_fraction':stat['depletion_fraction'],'switches':stat['switches'],
          'mac_per_recording':float(T*dense_mac_per_t()[0]+stat['mean_calls']*dense_mac_per_t()[1])}
        rows.append(row);return row
    add('low',{},0.,np.zeros_like(fixed_mask),{'mean_calls':0.,'mean_spill':0.,'depletion_fraction':0.,'switches':0.})
    add('fixed',{},0.,fixed_mask,fixed_stat)
    grids={'instant':[{}], 'ema':[{'rho':r} for r in (.7,.9)],
       'pirc':[{'rho':r,'alpha':alpha,'kappa':k,'eta':1.5}
          for r,alpha,k in itertools.product((.7,.9),(.25,.75),(4.,8.))]}
    for method,grid in grids.items():
        eligible=[]
        for params in grid:
            th,mask,stat=threshold_for_rate(rate,method,target,scenario,params)
            row=add(method,params,th,mask,stat)
            if abs(stat['mean_calls']-target)<=.3:eligible.append((row,params))
        if not eligible:raise RuntimeError(f'{method} cannot match fixed budget')
        best,params=max(eligible,key=lambda pair:(pair[0]['accuracy'],-pair[0]['ce']))
        chosen[method]={'params':params,'threshold':best['threshold'],
                        'val_accuracy':best['accuracy'],'val_calls':best['mean_calls']}
        print('SELECT',method,json.dumps(chosen[method]),flush=True)
    # Preserve PIRC gate parameters while removing only adaptive supply allocation.
    p=chosen['pirc'];th,mask,stat=threshold_for_rate(rate,'pirc_uniform_gate',target,scenario,p['params'])
    add('pirc_uniform_gate',p['params'],th,mask,stat)
    chosen['pirc_uniform_gate']={'params':p['params'],'threshold':float(th)}
    import csv
    with open(out/'validation_candidates.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
    locked={'checkpoint':a.checkpoint,'data':a.data,'val_indices':val.tolist(),
            'scenario':scenario,'fixed_target_calls':target,'chosen':chosen,
            'test_labels_unopened':True}
    (out/'locked.json').write_text(json.dumps(locked,indent=2))
    print('UTILITY',json.dumps(utility),flush=True)
    print('LOCKED',str(out/'locked.json'),flush=True)
if __name__=='__main__':main()
