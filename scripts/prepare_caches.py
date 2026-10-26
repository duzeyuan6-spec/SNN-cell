"""Cache frozen Backbone outputs for the release experiment without test-label tuning."""
import argparse
from pathlib import Path
import numpy as np
import torch
from local_pirc.select_controllers import cache, load_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=('fit','validation','official_test'),required=True)
    p.add_argument('--checkpoint',default='results/local_seed42/best.pt')
    p.add_argument('--data',default='data/prepared')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch',type=int,default=16)
    a=p.parse_args()
    torch.set_num_threads(4)
    model,ck=load_model(a.checkpoint,a.device)
    root=Path(a.data)
    if a.split=='fit':
        ids=np.asarray(ck['fit_indices'],dtype=np.int64)
        name='train'
        dest=Path('results/unified_followup/fit_cache.npz')
    elif a.split=='validation':
        ids=np.asarray(ck['val_indices'],dtype=np.int64)
        name='train'
        dest=Path('results/local_seed42/controllers/validation_cache.npz')
    else:
        name='test'
        ids=np.arange(len(np.load(root/'test_labels.npy')))
        dest=Path('results/local_seed42/controllers/final/test_cache.npz')
    assert not dest.exists(),f'Refuse overwrite: {dest}'
    frames=np.load(root/f'{name}_frames.npy',mmap_mode='r')
    labels=np.load(root/f'{name}_labels.npy')
    assert len(ids) and ids.min()>=0 and ids.max()<len(labels)
    low,corr,rate=cache(model,frames,ids,a.device,batch=a.batch)
    dest.parent.mkdir(parents=True,exist_ok=True)
    values={'low':low,'corr':corr,'rate':rate}
    if a.split!='official_test':
        values.update(indices=ids,labels=labels[ids])
    np.savez_compressed(dest,**values)
    if a.split=='fit':
        rate_path=Path('results/longrun_seed42/fit_rates.npy')
        assert not rate_path.exists(),f'Refuse overwrite: {rate_path}'
        rate_path.parent.mkdir(parents=True,exist_ok=True)
        np.save(rate_path,rate)
    print(f'SAVED {a.split}: {dest}, n={len(ids)}',flush=True)
if __name__=='__main__': main()
