"""Convert original SpikingJelly event samples to equal-duration ON/OFF bins."""
import argparse,json,re
from pathlib import Path
import numpy as np
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

T=16
H=64


def bin_sample(events):
    times=np.asarray(events['t'],dtype=np.int64)
    x=np.asarray(events['x'],dtype=np.int64)
    y=np.asarray(events['y'],dtype=np.int64)
    p=np.asarray(events['p'],dtype=np.int64)
    assert len(times)==len(x)==len(y)==len(p) and len(times)>0
    assert np.all((x>=0)&(x<128)&(y>=0)&(y<128)&(p>=0)&(p<2))
    t0=int(times.min());duration=max(1,int(times.max())-t0+1)
    tb=np.minimum(T-1,(times-t0)*T//duration)
    xx=x//2;yy=y//2
    flat=(((tb*2+p)*H+yy)*H+xx)
    counts=np.bincount(flat,minlength=T*2*H*H).reshape(T,2,H,H)
    assert counts.sum()==len(times)
    assert counts.max()<=np.iinfo(np.uint16).max
    frame=counts.astype(np.uint16)
    quadrants=frame.reshape(T,2,2,32,2,32).sum((1,3,5)).reshape(T,4)
    return frame,quadrants.astype(np.int32),duration


def one_split(root,out,train):
    ds=DVS128Gesture(root=str(root),train=train,data_type='event')
    n=len(ds);name='train' if train else 'test'
    frames=np.lib.format.open_memmap(out/f'{name}_frames.npy',mode='w+',dtype=np.uint16,
                                     shape=(n,T,2,H,H))
    quad=np.zeros((n,T,4),np.int32);labels=np.empty(n,np.int64);duration=np.empty(n,np.int64)
    sources=[]
    paths=getattr(ds,'samples',None)
    for i in range(n):
        event,label=ds[i]
        frame,q,d=bin_sample(event)
        frames[i]=frame;quad[i]=q;duration[i]=d;labels[i]=int(label)
        sources.append(str(paths[i][0]) if paths is not None else '')
        if (i+1)%100==0:print(name,i+1,'/',n,flush=True)
    frames.flush();del frames
    np.save(out/f'{name}_quadrants.npy',quad)
    np.save(out/f'{name}_labels.npy',labels)
    np.save(out/f'{name}_durations_us.npy',duration)
    (out/f'{name}_sources.json').write_text(json.dumps(sources,indent=2))
    return {'split':name,'n':n,'class_counts':np.bincount(labels,minlength=11).tolist(),
            'duration_us_quantiles':np.quantile(duration,[0,.25,.5,.75,1]).tolist(),
            'bin_count_cv':float(quad.sum(2).std()/quad.sum(2).mean()),
            'quadrant_count_cv':float(quad.std()/quad.mean()),
            'nonzero_quadrant_fraction':float(np.mean(quad>0))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='data/dvsgesture')
    ap.add_argument('--output',default='data/prepared')
    ap.add_argument('--split',choices=['train','test'],default='train');a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    stats=one_split(Path(a.root),out,a.split=='train')
    (out/f'{a.split}_metadata.json').write_text(json.dumps({'T':T,'H':H,
       'split_method':'original trial lists',
       'time_binning':'equal duration within each gesture; not equal event counts',
       'stats':stats},indent=2))
    print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
