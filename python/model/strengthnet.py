import math
import numpy as np
import torch
from torch import nn

class StrengthNet(nn.Module):
    def __init__(self, featureDims: int, hiddenDims: int=32):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(featureDims, hiddenDims),
            nn.ReLU()
        )
        self.rating = nn.Linear(hiddenDims, 1)
        self.weights = nn.Linear(hiddenDims, 1)
        self.softmax = nn.Softmax(dim=0)
        self.SCALE = 400 / math.log(10)  # Scale outputs to Elo/Glicko-like numbers

    def forward(self, x, xlens = None):
        # xlens specifies length of manually packed sequences in the batch
        if xlens is not None:
            clens = np.cumsum([0] + xlens)
        else:
            clens = [0, len(x)]  # assume one input (eg in evaluation)
        h = self.layer1(x)
        r = self.rating(h).squeeze(-1)
        z = self.weights(h).squeeze(-1)

        # predict one rating for each part in the batch
        parts = zip(clens[:-1], clens[1:])
        preds = [self._sumBySoftmax(r, z, start, end) for start, end in parts]
        return self.SCALE * torch.stack(preds)

    def _sumBySoftmax(self, r, z, start, end):
        if start == end:
            DEFAULT_PRED = 7.6699353278706015  # default prediction = 1332.40 Glicko
            return torch.tensor(DEFAULT_PRED, device=r.device)
        rslice = r[start:end]
        zslice = z[start:end]
        zslice = self.softmax(zslice)
        return torch.sum(zslice * rslice)
