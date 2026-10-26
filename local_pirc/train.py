"""Train one shared Conv-LIF Backbone and sparse local refinement branch."""
import argparse,json,random,re,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from .model import LocalSNN,T,GROUPS,dense_mac_per_t
from .controller import training_masks


class Frames(Dataset):
    def __init__(self,root,indices):
        self.frames=np.load(Path(root)/'train_frames.npy',mmap_mode='r')
        self.labels=np.load(Path(root)/'train_labels.npy')
        self.indices=np.asarray(indices)
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        ix=self.indices[i]
        return torch.from_numpy(np.array(self.frames[ix],copy=True)),int(self.labels[ix])


def split(root,seed):
    root=Path(root);labels=np.load(root/'train_labels.npy')
    paths=json.loads((root/'train_sources.json').read_text())
    subjects=[]
    for path in paths:
        m=re.search(r'user(\d+)',path)
        subjects.append(m.group(1) if m else None)
    rng=np.random.default_rng(seed)
    if all(s is not None for s in subjects):
        unique=np.array(sorted(set(subjects)));rng.shuffle(unique)
        val_subjects=set(unique[:max(1,round(.2*len(unique)))])
        val=np.array([i for i,s in enumerate(subjects) if s in val_subjects]);fit=np.array([i for i,s in enumerate(subjects) if s not in val_subjects])
        if len(set(labels[val]))==11 and len(set(labels[fit]))==11:
            method='subject-held-out';return fit,val,{'method':method,'val_subjects':sorted(val_subjects)}
    fit=[];val=[]
    for c in np.unique(labels):
        ix=np.flatnonzero(labels==c);rng.shuffle(ix);cut=max(1,round(.2*len(ix)))
        val.extend(ix[:cut]);fit.extend(ix[cut:])
    fit=np.asarray(fit);val=np.asarray(val);rng.shuffle(fit);rng.shuffle(val)
    return fit,val,{'method':'stratified-recording','val_subjects':[]}


@torch.no_grad()
def evaluate(model,loader,device,mask_seed=0):
    model.eval();loss_sum=0.;n=0;low_ok=0;high_ok=0;calls=0
    for x,y in loader:
        x=x.to(device,non_blocking=True);y=y.to(device)
        mask=torch.as_tensor(training_masks(len(x),mask_seed),device=device)
        low,high,_,nc=model(x,mask)
        loss_sum+=float(F.cross_entropy(high,y,reduction='sum'));n+=len(y)
        low_ok+=int((low.argmax(1)==y).sum());high_ok+=int((high.argmax(1)==y).sum())
        calls+=nc;model.reset()
    return {'high_ce':loss_sum/n,'low_acc':low_ok/n,'high_acc':high_ok/n,'mean_calls':calls/n}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/prepared')
    ap.add_argument('--out',default='results/local_seed42')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--split-file',default=None)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(4);device=torch.device(a.device)
    if a.split_file:
        predefined=json.loads(Path(a.split_file).read_text())
        fit=np.asarray(predefined['fit_indices']);val=np.asarray(predefined['val_indices'])
        split_meta={'method':'predeclared-subject-fold','split_file':a.split_file,
                    'val_subjects':predefined.get('val_subjects',[])}
        assert len(set(fit)&set(val))==0
        assert len(fit)>0 and len(val)>0
    else:
        fit,val,split_meta=split(a.data,a.seed)
    (out/'split.json').write_text(json.dumps({'fit_indices':fit.tolist(),'val_indices':val.tolist(),**split_meta},indent=2))
    train_loader=DataLoader(Frames(a.data,fit),batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True)
    val_loader=DataLoader(Frames(a.data,val),batch_size=a.batch_size,shuffle=False,num_workers=2,pin_memory=True)
    model=LocalSNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    (out/'config.json').write_text(json.dumps(vars(a)|{'fit_n':len(fit),'val_n':len(val),'split':split_meta,
      'dense_mac_per_t':dense_mac_per_t()},indent=2))
    best=float('inf');stale=0;start=time.time()
    with open(out/'training.jsonl','w') as log:
        for epoch in range(1,a.warmup+a.epochs+1):
            model.train();loss_total=0.;n=0;calls=0
            for batch,(x,y) in enumerate(train_loader):
                x=x.to(device,non_blocking=True);y=y.to(device)
                opt.zero_grad(set_to_none=True)
                if epoch<=a.warmup:
                    low,_,_,nc=model(x,None);loss=F.cross_entropy(low,y)
                else:
                    mask=torch.as_tensor(training_masks(len(x),a.seed+epoch+batch),device=device)
                    low,high,_,nc=model(x,mask)
                    loss=.4*F.cross_entropy(low,y)+.6*F.cross_entropy(high,y)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step();model.reset()
                loss_total+=float(loss.detach())*len(y);n+=len(y);calls+=nc
            metric=evaluate(model,val_loader,device)
            row={'epoch':epoch,'phase':'warmup' if epoch<=a.warmup else 'joint',
                 'train_loss':loss_total/n,'train_mean_calls':calls/n,
                 'val':metric,'elapsed_s':round(time.time()-start,1)}
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if epoch>a.warmup:
                if metric['high_ce']<best-1e-4:
                    best=metric['high_ce'];stale=0
                    torch.save({'model':model.state_dict(),'seed':a.seed,'epoch':epoch,
                       'fit_indices':fit,'val_indices':val,'config':vars(a),
                       'val_metric':metric},out/'best.pt')
                else:
                    stale+=1
                    if stale>=6:
                        print('EARLY_STOP',epoch,flush=True);break
    print('BEST',best,'CHECKPOINT',out/'best.pt',flush=True)
if __name__=='__main__':main()
