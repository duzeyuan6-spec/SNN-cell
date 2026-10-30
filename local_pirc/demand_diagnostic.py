"""Causal reserve-only versus reserve-plus-spike-history shortage diagnostic."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from .model import T
from .outage_sweep import SCENARIOS, DELAY, stream_inputs, simulate
from .predictive_online import save_csv

FIT_SEEDS=(701,702,703)
VAL_SEEDS=tuple(range(711,721))
HORIZONS=(16,32)
GAINS=(0.,1.,2.,4.)
REF={'gain':2.,'boost':4.,'level':12.}


def activity_history(activity):
    """Current and past-only firing-rate features; clear neural history per gesture."""
    n,g=activity.shape
    out=np.empty((n,g,4),dtype=np.float32)
    e4=np.zeros(g);e16=np.zeros(g)
    for t in range(n):
        if t%T==0:e4[:]=0.;e16[:]=0.
        a=activity[t]
        e4=.75*e4+.25*a
        e16=.9375*e16+.0625*a
        out[t,:,0]=a
        out[t,:,1]=e4
        out[t,:,2]=e16
        out[t,:,3]=e4-e16
    return out


def feature_matrix(reserve,history,kind):
    r=np.asarray(reserve,dtype=np.float64)
    x=[r/12.,np.maximum(0.,1-r/2.),np.maximum(0.,1-r/4.),np.maximum(0.,1-r/8.)]
    if kind=='activity':x.extend([np.asarray(history[...,i],dtype=np.float64) for i in range(4)])
    elif kind!='reserve':raise ValueError(kind)
    return np.stack(x,axis=-1)


def future_shortage(denied,horizon):
    assert horizon>=DELAY
    n=len(denied)
    cs=np.vstack((np.zeros((1,denied.shape[1]),dtype=np.int32),np.cumsum(denied.astype(np.int32),axis=0)))
    t=np.arange(n-horizon)
    return (cs[t+horizon+1]-cs[t+DELAY]>0)


class RiskModel:
    def __init__(self,kind,mean,scale,coef):
        self.kind=kind;self.mean=np.asarray(mean);self.scale=np.asarray(scale);self.coef=np.asarray(coef)
    def predict(self,reserve,history):
        x=feature_matrix(reserve,history,self.kind)
        z=(x-self.mean)/self.scale
        return expit(z@self.coef[:-1]+self.coef[-1])
    def to_dict(self):return {'kind':self.kind,'mean':self.mean.tolist(),'scale':self.scale.tolist(),'coef':self.coef.tolist()}
    @classmethod
    def from_dict(cls,d):return cls(d['kind'],d['mean'],d['scale'],d['coef'])


def fit_model(x,y,kind):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    mean=x.mean(0);scale=x.std(0);scale=np.maximum(scale,1e-5)
    z=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    reg=1e-3
    def objective(w):
        q=z@w
        loss=np.logaddexp(0,q).mean()-np.dot(y,q)/len(y)+.5*reg*np.dot(w[:-1],w[:-1])
        grad=z.T@(expit(q)-y)/len(y)
        grad[:-1]+=reg*w[:-1]
        return loss,grad
    initial=np.zeros(z.shape[1]);initial[-1]=np.log(np.mean(y)/(1-np.mean(y)))
    fit=minimize(objective,initial,method='L-BFGS-B',jac=True,options={'maxiter':100,'ftol':1e-9})
    assert fit.success or np.max(np.abs(fit.jac))<1e-4,(fit.message,fit.nit)
    return RiskModel(kind,mean,scale,fit.x),{'iterations':int(fit.nit),'fit_loss':float(fit.fun),'prevalence':float(y.mean())}


def classification_metrics(y,p):
    y=np.asarray(y,dtype=np.int8);p=np.clip(np.asarray(p,dtype=np.float64),1e-9,1-1e-9)
    positives=int(y.sum());negative=len(y)-positives
    rank=rankdata(p,method='average')
    auc=(rank[y==1].sum()-positives*(positives+1)/2)/(positives*negative)
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    ends=np.r_[np.flatnonzero(ps[1:]!=ps[:-1]),len(ps)-1]
    tp=np.cumsum(ys)[ends];precision=tp/(ends+1);recall=tp/positives
    ap=float(np.sum(precision*np.diff(np.r_[0,recall])))
    return {'auroc':float(auc),'auprc':ap,'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
            'brier':float(np.mean((p-y)**2)),'prevalence':positives/len(y)}


def trace_data(rates,scenario,seed):
    inp=stream_inputs(rates,scenario,seed)
    _,_,trace=simulate(inp,'reserve_threshold',**REF)
    return inp,trace,activity_history(inp['activity'])


def build_xy(trace,hist,horizon,kind):
    target=future_shortage(trace['denied'],horizon)
    n=len(target)
    x=feature_matrix(trace['reserve'][:n],hist[:n],kind).reshape(-1,4 if kind=='reserve' else 8)
    return x,target.reshape(-1)


def mean_scenario(rows,metric,method):
    return float(np.mean([r[metric] for r in rows if r['method']==method]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/demand_diagnostic_seed42');a=ap.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'validation_prediction.csv').exists(),'Diagnostic already complete; refuse overwrite'
    fitrates=np.load('results/longrun_seed42/fit_rates.npy')
    valrates=np.load('results/local_seed42/controllers/validation_cache.npz')['rate']
    assert fitrates.shape==(924,16,4) and valrates.shape==(252,16,4)
    fit_traces=[]
    for scenario in SCENARIOS:
        for seed in FIT_SEEDS:
            fit_traces.append((scenario,seed,*trace_data(fitrates,scenario,seed)))
        print('FIT_TRACES',scenario,flush=True)
    models={};fitinfo=[]
    for horizon in HORIZONS:
        for kind in ('reserve','activity'):
            arrays=[build_xy(trace,hist,horizon,kind) for _,_,_,trace,hist in fit_traces]
            x=np.concatenate([a[0] for a in arrays]);y=np.concatenate([a[1] for a in arrays])
            model,info=fit_model(x,y,kind)
            models[(horizon,kind)]=model
            fitinfo.append({'horizon':horizon,'method':kind,'n_rows':len(y),**info})
            print('FIT_MODEL',horizon,kind,info,flush=True)
    save_csv(root/'fit_models.csv',fitinfo)
    (root/'models.json').write_text(json.dumps({f'{h}_{k}':v.to_dict() for (h,k),v in models.items()},indent=2))
    # Calibrate the deployable gain using fit traces only; the objective is fixed in protocol.
    fit_policy=[];choices={}
    for kind in ('reserve','activity'):
        for risk_gain in GAINS:
            vals=[]
            for scenario,seed,inp,_,hist in fit_traces:
                st=simulate(inp,'risk_'+kind,return_masks=False,risk_model=models[(16,kind)],
                            activity_features=hist,risk_gain=risk_gain,**REF)
                vals.append(st)
            obj=np.mean([.5*s['outage_rcir']+.5*s['rcir']+.1*s['spill_fraction'] for s in vals])
            fit_policy.append({'method':kind,'risk_gain':risk_gain,'objective':float(obj),
                'outage_service':float(np.mean([s['outage_service'] for s in vals])),
                'rcir':float(np.mean([s['rcir'] for s in vals]))})
            print('FIT_POLICY',kind,risk_gain,obj,flush=True)
        choices[kind]=min((r for r in fit_policy if r['method']==kind),key=lambda r:(r['objective'],r['risk_gain']))['risk_gain']
    save_csv(root/'fit_policy.csv',fit_policy)
    (root/'choices.json').write_text(json.dumps(choices,indent=2))
    pred_rows=[];policy_rows=[]
    for scenario in SCENARIOS:
        for seed in VAL_SEEDS:
            inp,trace,hist=trace_data(valrates,scenario,seed)
            for horizon in HORIZONS:
                for kind in ('reserve','activity'):
                    x,y=build_xy(trace,hist,horizon,kind)
                    p=models[(horizon,kind)].predict(trace['reserve'][:len(trace['reserve'])-horizon],hist[:len(hist)-horizon]).reshape(-1)
                    assert len(p)==len(y)
                    pred_rows.append({'scenario':scenario,'seed':seed,'horizon':horizon,'method':kind,'n_rows':len(y),**classification_metrics(y,p)})
            for method in ('reserve_threshold','risk_reserve','risk_activity'):
                if method=='reserve_threshold':params={}
                else:
                    kind=method.removeprefix('risk_')
                    params={'risk_model':models[(16,kind)],'risk_gain':choices[kind],'activity_features':hist}
                mask,st,_=simulate(inp,method,**REF,**params)
                policy_rows.append({'scenario':scenario,'seed':seed,'method':method,
                    'outage_service':st['outage_service'],'rcir':st['rcir'],'service_rate':st['service_rate'],
                    'spill_fraction':st['spill_fraction'],'executed':st['executed'],
                    'high_calls_per_clip':st['executed']/st['nclips'],
                    'mac_per_recording':st['mac_per_recording'],
                    'preoutage_reserve':st['preoutage_reserve']})
        print('VALIDATION',scenario,'done',flush=True)
    save_csv(root/'validation_prediction.csv',pred_rows)
    save_csv(root/'validation_policy.csv',policy_rows)
    (root/'manifest.json').write_text(json.dumps({'scenarios':SCENARIOS,'fit_seeds':FIT_SEEDS,'validation_seeds':VAL_SEEDS,
        'horizons':HORIZONS,'reference':REF,'choices':choices,'checkpoint':'results/local_seed42/best.pt',
        'official_test_used':False,'checkpoint_retrained':False},indent=2))
    print('PREDICTION_EQUAL_SCENARIO')
    for horizon in HORIZONS:
        for kind in ('reserve','activity'):
            sub=[r for r in pred_rows if r['horizon']==horizon and r['method']==kind]
            print(horizon,kind,{m:float(np.mean([r[m] for r in sub])) for m in ('auroc','auprc','log_loss','brier','prevalence')})
    print('POLICY_EQUAL_SCENARIO')
    for method in ('reserve_threshold','risk_reserve','risk_activity'):
        sub=[r for r in policy_rows if r['method']==method]
        print(method,{m:float(np.mean([r[m] for r in sub])) for m in ('outage_service','rcir','service_rate','spill_fraction','mac_per_recording')})

if __name__=='__main__':main()
