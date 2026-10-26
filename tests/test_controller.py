import numpy as np
from local_pirc.controller import schedule,training_masks,T,G


def test_resource_conservation_and_causality():
    rng=np.random.default_rng(4);a=rng.random((8,T,G))*.2
    b=a.copy();b[:,9:]=1
    for method in ('fixed','instant','ema','pirc','pirc_uniform_gate'):
        x,s=schedule(a,method,threshold=.1)
        y,_=schedule(b,method,threshold=.1)
        assert np.array_equal(x[:,:9],y[:,:9])
        assert np.all(s['reserve']>=-1e-8)
        assert np.allclose(s['supply'].sum(2),[1.]*4+[.5]*8+[1.]*4)
        assert np.all(s['spill']>=0)


def test_random_training_masks_shape():
    for seed in (0,1):
        m=training_masks(7,seed)
        assert m.shape==(7,T,G) and m.dtype==bool
