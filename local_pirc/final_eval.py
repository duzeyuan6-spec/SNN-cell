"""One-shot official-test evaluation after validation parameters are locked."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from .model import T,dense_mac_per_t
from .controller import schedule
from .select_controllers import load_model,cache,score


def save_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)


@torch.no_grad()
def actual_sparse(model,frames,mask,device,batch=16,return_logits=False):
    model.eval();pred=[];logits=[];total_ms=0.;calls=0
    for start in range(0,len(mask),batch):
        end=min(len(mask),start+batch)
        x=torch.as_tensor(np.array(frames[start:end],copy=True),device=device)
        m=torch.as_tensor(mask[start:end],device=device)
        torch.cuda.synchronize()
        first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        first.record();low,high,_,nc=model(x,m);last.record();torch.cuda.synchronize()
        total_ms+=first.elapsed_time(last);calls+=nc
        output=high if high is not None else low
        pred.extend(output.argmax(1).cpu().tolist())
        if return_logits:logits.append(output.cpu().numpy())
        model.reset()
    assert calls==int(mask.sum())
    return (np.concatenate(logits) if return_logits else np.asarray(pred)),total_ms/len(mask)


def paired_bootstrap(predictions,y):
    rng=np.random.default_rng(6868);n=len(y);draw=rng.integers(0,n,size=(5000,n))
    base=(predictions['pirc']==y).astype(float)
    out={}
    for name,pred in predictions.items():
        d=(pred==y).astype(float)-base
        out[name]={'accuracy_minus_pirc':float(d.mean()),
                   'ci95':np.quantile(d[draw].mean(1),[.025,.975]).tolist()}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--locked',default='results/local_seed42/controllers/locked.json')
    ap.add_argument('--output',default='results/local_seed42/controllers/final')
    ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'results.csv').exists(), 'Final test already evaluated; do not iterate on labels.'
    lock=json.loads(Path(a.locked).read_text())
    torch.set_num_threads(4);model,_=load_model(lock['checkpoint'],a.device)
    root=Path(lock['data']);frames=np.load(root/'test_frames.npy',mmap_mode='r')
    y=np.load(root/'test_labels.npy')
    test_indices=np.arange(len(y));low,corr,rate=cache(model,frames,test_indices,a.device)
    np.savez_compressed(out/'test_cache.npz',low=low,corr=corr,rate=rate)
    methods=['low','fixed','instant','ema','pirc','pirc_uniform_gate']
    base,extra=dense_mac_per_t();rows=[];sample=[];predictions={}
    for method in methods:
        tick=time.perf_counter()
        if method=='low':
            mask=np.zeros((len(y),T,4),bool);stat={'mean_calls':0.,'mean_spill':0.,
                  'depletion_fraction':0.,'switches':0.}
        elif method=='fixed':mask,stat=schedule(rate,'fixed',scenario=lock['scenario'])
        else:
            s=lock['chosen'][method]
            mask,stat=schedule(rate,method,threshold=s['threshold'],
                               scenario=lock['scenario'],**s['params'])
        controller_ms=(time.perf_counter()-tick)*1000/len(y)
        q=score(low,corr,y,mask)
        # Warm up then time actual physically skipped convolution path four times.
        actual_sparse(model,frames,mask,a.device)
        trials=[]
        for _ in range(4):
            p,ms=actual_sparse(model,frames,mask,a.device)
            assert float(np.mean(p==y))==q['accuracy']
            trials.append(ms)
        predictions[method]=p
        rows.append({'method':method,'n':len(y),'accuracy':q['accuracy'],
          'cross_entropy':q['ce'],'mean_calls':stat['mean_calls'],
          'trigger_rate':stat['mean_calls']/(T*4),
          'dense_mac_per_recording':T*base+stat['mean_calls']*extra,
          'dense_mac_per_bin':base+stat['mean_calls']*extra/T,
          'budget_violation_rate':0.,'mean_spill':stat['mean_spill'],
          'depletion_fraction':stat['depletion_fraction'],
          'switches_per_recording':stat['switches'],
          'gpu_network_ms_per_recording':float(np.median(trials)),
          'gpu_latency_min_ms':float(min(trials)),'gpu_latency_max_ms':float(max(trials)),
          'cpu_controller_ms_per_recording':controller_ms})
        print('FINAL',json.dumps(rows[-1]),flush=True)
    for i in range(len(y)):
        r={'test_index':i,'label':int(y[i])}
        for method in methods:r[method+'_pred']=int(predictions[method][i])
        sample.append(r)
    save_csv(out/'results.csv',rows);save_csv(out/'predictions.csv',sample)
    (out/'paired_bootstrap.json').write_text(json.dumps(paired_bootstrap(predictions,y),indent=2))
    print('SAVED',out,flush=True)
if __name__=='__main__':main()
