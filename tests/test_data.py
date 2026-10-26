import numpy as np
from local_pirc.prepare_data import bin_sample


def test_equal_duration_preserves_density_and_boundaries():
    e={'t':np.array([0,0,1,1,15,15,15]),'x':np.array([0,0,10,10,127,127,127]),
       'y':np.array([0,0,10,10,127,127,127]),'p':np.array([0,1,0,0,1,1,1])}
    f,q,d=bin_sample(e)
    assert f.sum()==7 and q.sum()==7 and d==16
    assert q[0].sum()==2 and q[-1].sum()==3
    assert q[2:15].sum()==0
