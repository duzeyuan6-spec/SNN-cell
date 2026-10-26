"""Causal local allocation with finite conserved supply and optional branch costs."""
import numpy as np

T=16
G=4
RMAX=2.
INITIAL=1.


def supply(t,scenario='bottleneck'):
    if scenario=='steady':return 1.
    if scenario=='bottleneck':return .5 if 4<=t<12 else 1.
    if scenario=='severe':return .25 if 4<=t<12 else 1.
    raise ValueError(scenario)


def _softmax(x):
    x=x-x.max(axis=1,keepdims=True)
    z=np.exp(x)
    return z/z.sum(axis=1,keepdims=True)


def schedule(activity,method,threshold=0.,rho=.8,alpha=.5,kappa=8.,eta=1.5,
             scenario='bottleneck',seed=42):
    """N,T,4 observed local spike rates -> feasible N,T,4 calls.

    Time t activity is measured from the *base* SNN before optional refinement.
    It can change calls at t, never the base spikes that produced it. State resets
    for each recording. All policies get the same exogenous global token supply.
    """
    activity=np.asarray(activity,dtype=np.float64)
    assert activity.ndim==3 and activity.shape[1:]==(T,G)
    assert np.all((activity>=0)&(activity<=1))
    n=len(activity);reserve=np.full((n,G),INITIAL);slow=np.zeros((n,G))
    gates=np.zeros((n,T,G),bool);reserves=np.zeros((n,T,G))
    allocs=np.zeros((n,T,G));spill=np.zeros((n,T,G));blocked=np.zeros((n,T,G),bool)
    rng=np.random.default_rng(seed)
    for t in range(T):
        d=activity[:,t,:]
        slow=rho*slow+(1-rho)*d
        B=supply(t,scenario)
        if method in ('pirc','pirc_uniform_gate'):
            score_supply=kappa*slow+eta*(1-reserve/RMAX)
            weights=_softmax(score_supply) if method=='pirc' else np.full((n,G),1/G)
        else:weights=np.full((n,G),1/G)
        incoming=B*weights
        assert np.allclose(incoming.sum(1),B)
        available=np.minimum(RMAX,reserve+incoming)
        spill[:,t,:]=np.maximum(0.,reserve+incoming-RMAX)
        allocs[:,t,:]=incoming
        if method=='fixed':
            requested=np.zeros((n,G),bool);requested[:,t%G]=True
        elif method=='random':
            requested=rng.random((n,G))<threshold
        elif method=='instant':requested=d>threshold
        elif method=='ema':requested=slow>threshold
        elif method in ('pirc','pirc_uniform_gate'):
            requested=d+alpha*slow+.2*(available/RMAX) > threshold
        else:raise ValueError(method)
        actual=requested & (available>=1.-1e-9)
        blocked[:,t,:]=requested & ~actual
        new_reserve=available-actual.astype(float)
        # Exact token accounting for every recording, including cap spill.
        assert np.allclose(reserve.sum(1)+B,
            new_reserve.sum(1)+actual.sum(1)+spill[:,t,:].sum(1),atol=1e-7)
        reserve=new_reserve
        gates[:,t,:]=actual;reserves[:,t,:]=reserve
    return gates,{'reserve':reserves,'supply':allocs,'spill':spill,'blocked':blocked,
                  'calls':gates.sum((1,2)),
                  'mean_calls':float(gates.sum((1,2)).mean()),
                  'mean_reserve':float(reserves.mean()),
                  'mean_spill':float(spill.sum((1,2)).mean()),
                  'depletion_fraction':float(np.mean(reserves<.01)),
                  'switches':float(np.mean(np.abs(np.diff(gates.astype(np.int8),axis=1)).sum((1,2))))}


def training_masks(n,seed,scenario='bottleneck'):
    rng=np.random.default_rng(seed)
    reserve=np.full((n,G),INITIAL);out=np.zeros((n,T,G),bool)
    for t in range(T):
        incoming=supply(t,scenario)/G
        available=np.minimum(RMAX,reserve+incoming)
        if seed%2==0:
            requested=np.zeros((n,G),bool);requested[:,t%G]=True
        else:requested=rng.random((n,G))<.3
        call=requested & (available>=1.-1e-9)
        reserve=available-call.astype(float);out[:,t,:]=call
    return out
