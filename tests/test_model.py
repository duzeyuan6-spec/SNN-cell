import torch
from local_pirc.model import LocalSNN,GROUPS


def test_optional_convolution_is_physically_skipped_and_gradients_flow():
    model=LocalSNN()
    local=torch.rand((2,32,16,16),requires_grad=True)
    called=[]
    hook=model.refine.conv1.register_forward_hook(lambda *args:called.append(True))
    zero,n=model.sparse_correction(local,torch.zeros((2,GROUPS),dtype=torch.bool))
    assert n==0 and called==[] and torch.all(zero==0)
    mask=torch.zeros((2,GROUPS),dtype=torch.bool);mask[0,2]=True
    out,n=model.sparse_correction(local,mask)
    assert n==1 and len(called)==1
    out.sum().backward()
    assert model.refine.conv1.weight.grad is not None
    assert local.grad is not None
    hook.remove()
