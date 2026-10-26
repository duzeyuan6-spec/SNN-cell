import torch
from local_pirc.compact_high import CompactLocalSNN, CompactHigh, compact_extra_mac
from local_pirc.model import dense_mac_per_t

def test_compact_features_use_current_state_only():
    local=torch.zeros(2,32,16,16)
    local[0,:,0:8,0:8]=1
    low=torch.arange(22,dtype=torch.float32).reshape(2,11)
    feat=CompactLocalSNN.compact_features(local,low)
    assert feat.shape==(2,4,527)
    assert torch.allclose(feat[0,0,-11:],low[0])
    assert torch.allclose(feat[1,3,-11:],low[1])
    changed=local.clone();changed[0,:,8:,8:]=1
    feat2=CompactLocalSNN.compact_features(changed,low)
    assert torch.equal(feat[0,0],feat2[0,0])
    assert not torch.equal(feat[0,3],feat2[0,3])

def test_zero_mask_skips_compact_head():
    model=CompactLocalSNN().eval()
    local=torch.zeros(2,32,16,16)
    mask=torch.zeros(2,4,dtype=torch.bool)
    partial=torch.zeros(2,11)
    out,n=model.sparse_correction_compact(local,mask,partial)
    assert n==0 and torch.count_nonzero(out)==0

def test_compact_extra_is_positive_and_smaller_than_original():
    old=dense_mac_per_t()[1]
    assert 0<compact_extra_mac()<old
