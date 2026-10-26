"""Three-stage Conv-LIF classifier with genuinely optional local refinement."""
import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import neuron,surrogate

T=16
GROUPS=4
CLASSES=11


def lif():
    return neuron.LIFNode(tau=2.,surrogate_function=surrogate.ATan(),detach_reset=True,step_mode='s')


class LocalRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(32,64,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.conv2=nn.Conv2d(64,64,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(64)
        self.region_embed=nn.Embedding(GROUPS,8)
        self.head=nn.Linear(72,CLASSES)

    def features(self,patch,region_id):
        h=F.relu(self.bn1(self.conv1(patch)))
        h=F.relu(self.bn2(self.conv2(h)))
        pooled=h.mean((2,3))
        return torch.cat((pooled,self.region_embed(region_id)),dim=1)

    def forward(self,patch,region_id):
        return self.head(self.features(patch,region_id))


class LocalSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(2,16,3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.lif1=lif()
        self.conv2=nn.Conv2d(16,32,3,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(32)
        self.lif2=lif()
        self.conv3=nn.Conv2d(32,64,3,padding=1,bias=False)
        self.bn3=nn.BatchNorm2d(64)
        self.lif3=lif()
        self.low_head=nn.Linear(64,CLASSES)
        self.refine=LocalRefinement()

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()
        self.lif3.reset()

    def step(self,frame):
        assert frame.ndim==4 and frame.shape[1:]==(2,64,64)
        h=torch.log1p(frame.float())/3.
        h=self.lif1(self.bn1(self.conv1(h)))
        h=F.avg_pool2d(h,2)
        h=self.lif2(self.bn2(self.conv2(h)))
        h=F.avg_pool2d(h,2) # local map B,32,16,16
        local=h
        h=self.lif3(self.bn3(self.conv3(h)))
        h=F.avg_pool2d(h,2)
        low=self.low_head(h.mean((2,3)))
        return low,local

    @staticmethod
    def patches(local):
        b,c,h,w=local.shape
        assert (c,h,w)==(32,16,16)
        return local.reshape(b,c,2,8,2,8).permute(0,2,4,1,3,5).reshape(b*GROUPS,c,8,8)

    @staticmethod
    def activity(local):
        b,c,h,w=local.shape
        return local.reshape(b,c,2,8,2,8).mean((1,3,5)).reshape(b,GROUPS)

    def sparse_correction(self,local,mask):
        """Only selected sample-region patches are processed by the optional convolutions."""
        b=len(local)
        assert mask.shape==(b,GROUPS)
        selected=torch.nonzero(mask.reshape(-1),as_tuple=False).squeeze(1)
        if selected.numel()==0:
            return local.new_zeros((b,CLASSES)),0
        patches=self.patches(local).index_select(0,selected)
        correction=self.refine(patches,selected%GROUPS)
        out=local.new_zeros((b,CLASSES))
        out.index_add_(0,selected//GROUPS,correction)
        return out,int(selected.numel())

    def forward(self,x,masks=None,record_activity=False):
        assert x.ndim==5 and x.shape[1:]==(T,2,64,64)
        b=len(x)
        if masks is not None:assert masks.shape==(b,T,GROUPS)
        low_sum=x.new_zeros((b,CLASSES),dtype=torch.float32)
        high_sum=x.new_zeros((b,CLASSES),dtype=torch.float32)
        activities=[];calls=0
        for t in range(T):
            low,local=self.step(x[:,t]);low_sum=low_sum+low
            if masks is not None:
                corr,n=self.sparse_correction(local,masks[:,t]);high_sum=high_sum+corr;calls+=n
            if record_activity:activities.append(self.activity(local).detach())
        low_logits=low_sum/T
        high_logits=(low_sum+high_sum)/T if masks is not None else None
        rates=torch.stack(activities,dim=1) if record_activity else None
        return low_logits,high_logits,rates,calls


def dense_mac_per_t():
    # One 64x64 input; 3x3 dense convolution and final Low linear.
    base=64*64*2*16*9+32*32*16*32*9+16*16*32*64*9+64*11
    extra=8*8*(32*64+64*64)*9+72*11
    return base,extra
