"""Low-MAC causal local correction candidate. The old backbone is frozen."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import LocalSNN, T, GROUPS, CLASSES, dense_mac_per_t

FEATURE_DIM = 32 * 16 + GROUPS + CLASSES
HIDDEN = 128

class CompactHigh(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURE_DIM, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, CLASSES))

    def forward(self, features):
        return self.net(features)

class CompactLocalSNN(LocalSNN):
    def __init__(self):
        super().__init__()
        self.compact = CompactHigh()

    @staticmethod
    def compact_features(local, cumulative_low):
        b = len(local)
        patch = LocalSNN.patches(local)
        pooled = F.avg_pool2d(patch, 2).flatten(1).reshape(b, GROUPS, 512)
        region = F.one_hot(torch.arange(GROUPS, device=local.device), GROUPS).to(pooled.dtype)
        region = region[None].expand(b, -1, -1)
        low = cumulative_low[:, None, :].expand(-1, GROUPS, -1)
        return torch.cat((pooled, region, low), dim=-1)

    def sparse_correction_compact(self, local, mask, cumulative_low):
        b = len(local)
        assert mask.shape == (b, GROUPS)
        selected = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(1)
        if selected.numel() == 0:
            return local.new_zeros((b, CLASSES)), 0
        # Selection occurs before the learned correction; no High work for unselected regions.
        patches = self.patches(local).index_select(0, selected)
        pooled = F.avg_pool2d(patches, 2).flatten(1)
        region = F.one_hot(selected % GROUPS, GROUPS).to(pooled.dtype)
        low = cumulative_low.index_select(0, selected // GROUPS)
        correction = self.compact(torch.cat((pooled, region, low), -1))
        out = local.new_zeros((b, CLASSES))
        out.index_add_(0, selected // GROUPS, correction)
        return out, int(selected.numel())

    def forward(self, x, masks=None, record_activity=False):
        assert x.ndim == 5 and x.shape[1:] == (T, 2, 64, 64)
        b = len(x)
        if masks is not None:
            assert masks.shape == (b, T, GROUPS)
        low_sum = x.new_zeros((b, CLASSES), dtype=torch.float32)
        high_sum = x.new_zeros((b, CLASSES), dtype=torch.float32)
        activities = []
        calls = 0
        for t in range(T):
            low, local = self.step(x[:, t])
            low_sum = low_sum + low
            if masks is not None:
                corr, n = self.sparse_correction_compact(local, masks[:, t], low_sum / (t + 1))
                high_sum += corr
                calls += n
            if record_activity:
                activities.append(self.activity(local).detach())
        rates = torch.stack(activities, 1) if record_activity else None
        return low_sum / T, (low_sum + high_sum) / T if masks is not None else None, rates, calls

def compact_extra_mac():
    # 4x4 spatial pooling per channel, two dense layers, and no High convolution.
    return 512 * 4 + FEATURE_DIM * HIDDEN + HIDDEN * CLASSES
