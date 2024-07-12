#!/usr/bin/env python3
# Our main model implementation.
# Code adapted from paper author's implementation: https://github.com/juho-lee/set_transformer

import math
import numpy as np
import torch
from torch import nn
import inspect

class StrengthNet(nn.Module):
    """Full model based on Set Transformer"""

    def __init__(self, featureDims: int, depth: int=3, hiddenDims: int=64, queryDims: int=64, inducingPoints: int=64):
        super(StrengthNet, self).__init__()

        layers = []
        layers.append(nn.Linear(featureDims, hiddenDims, bias=False))
        for _ in range(depth):
            layers.append(ISAB(hiddenDims, queryDims, inducingPoints))
        self.enc = Sequential(*layers)
        self.dec = Sequential(
                Pool(hiddenDims, queryDims),
                nn.Linear(hiddenDims, 1, bias=False))

    def forward(self, x, xlens = None):
        return self.dec(self.enc(x, xlens), xlens).squeeze(-1)

    def activations(self):
        """Get flat values by layer from last forward pass for introspection"""
        a_acts = []
        h_acts = []
        for layer in self.enc.layers:
            if isinstance(layer, ISAB):
                a = torch.cat((layer.ab0.a.flatten(), layer.ab1.a.flatten()))
                a_acts.append(a)
                hres = torch.cat((layer.ab0.hres.flatten(), layer.ab1.hres.flatten()))
                h_acts.append(hres)
        return a_acts, h_acts

class Sequential(nn.Module):
    """Like nn.Sequential, but passes xlens (collated minibatch structure) where necessary"""
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, xlens = None):
        for layer in self.layers:
            sig = inspect.signature(layer.forward)
            if 'xlens' in sig.parameters:
                x = layer(x, xlens)
            else:
                x = layer(x)
        return x

class AttentionBlock(nn.Module):
    """Dot-product attention + FC + norms; no batching"""
    def __init__(self, hiddenDims: int=64, queryDims: int=64):
        super(AttentionBlock, self).__init__()
        self.z = torch.sqrt(torch.tensor(queryDims, dtype=torch.float32))
        self.WQ = nn.Linear(hiddenDims, queryDims, bias=False)
        self.WK = nn.Linear(hiddenDims, queryDims, bias=False)
        self.WV = nn.Linear(hiddenDims, hiddenDims, bias=False)
        self.fc = nn.Linear(hiddenDims, hiddenDims, bias=True)
        self.norm0 = nn.LayerNorm(hiddenDims)
        self.norm1 = nn.LayerNorm(hiddenDims)

    def forward(self, q, h):
        qq = self.WQ(q)
        k, v = self.WK(h), self.WV(h)

        a = torch.matmul(qq, k.transpose(-2, -1)) / self.z
        self.a = torch.softmax(a, dim=-1)  # store activations for introspection
        h = self.norm0(q + torch.matmul(self.a, v))
        self.hres = self.fc(h)  # store preactivations for introspection
        h = self.norm1(h + torch.relu(self.hres))
        return h

class ISAB(nn.Module):
    """Induced set attention block"""
    def __init__(self, hiddenDims: int=64, queryDims: int=64, inducingPoints: int=64):
        super(ISAB, self).__init__()
        self.i = nn.Parameter(torch.Tensor(inducingPoints, hiddenDims))
        nn.init.uniform_(self.i)
        self.ab0 = AttentionBlock(hiddenDims, queryDims)
        self.ab1 = AttentionBlock(hiddenDims, queryDims)

    def forward(self, x, xlens = None):
        os = []  # outputs

        for start, end in unbatch(x, xlens):
            h = x[start:end]
            i = self.ab0(self.i, h)
            o = self.ab1(h, i)
            os.append(o)

        os = torch.cat(os, dim=0)
        return os

class Pool(nn.Module):
    """Attention pooling layer with learned seed vector"""
    def __init__(self, hiddenDims: int=64, queryDims: int=64, inducingPoints: int=64):
        super(Pool, self).__init__()
        self.s = nn.Parameter(torch.Tensor(1, hiddenDims))
        nn.init.uniform_(self.s)
        self.ab = AttentionBlock(hiddenDims, queryDims)

    def forward(self, x, xlens = None):
        os = []  # outputs

        for start, end in unbatch(x, xlens):
            h = x[start:end]
            o = self.ab(self.s, h)
            os.append(o)

        os = torch.cat(os, dim=0)
        return os

def unbatch(x, xlens):
    """Get an iterable over the batch elements in x"""
    # xlens specifies length of manually packed sequences in the batch
    if xlens:
        clens = np.cumsum([0] + xlens)
    else:
        clens = [0, len(x)]  # assume one input (eg in evaluation)
    return zip(clens[:-1], clens[1:])

class PocStrengthNet(nn.Module):
    """Proof of concept variant"""
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
