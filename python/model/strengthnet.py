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

    def __init__(self, featureDims: int, depth: int, hiddenDims: int,
                 queryDims: int, inducingPoints: int, introspection: bool = False):
        super(StrengthNet, self).__init__()
        self.featureDims = featureDims
        self.depth = depth
        self.hiddenDims = hiddenDims
        self.queryDims = queryDims
        self.inducingPoints = inducingPoints
        self.introspection = introspection

        layers = []
        layers.append(nn.Linear(featureDims, hiddenDims, bias=False))
        for _ in range(depth):
            layers.append(ISAB(hiddenDims, queryDims, inducingPoints, introspection=introspection))
        self.enc = Sequential(*layers, introspection=introspection)
        self.dec = Sequential(
                Pool(hiddenDims, queryDims, introspection=introspection),
                nn.Linear(hiddenDims, 1, bias=False),
                introspection=introspection)

    def forward(self, x, xlens = None):
        return self.dec(self.enc(x, xlens), xlens).squeeze(-1)

    def embeddings(self):
        """Get outputs by layer from last forward pass for introspection"""
        return self.enc.hs + self.dec.hs[:-1]

    def activations(self):
        """Get flat values by layer from last forward pass for introspection"""
        a_acts = []
        h_acts = []
        for layer in self.enc.layers:
            if isinstance(layer, ISAB):
                a = torch.cat((layer.ab0.at.a.flatten(), layer.ab1.at.a.flatten()))
                a_acts.append(a)
                hres = torch.cat((layer.ab0.hres.flatten(), layer.ab1.hres.flatten()))
                h_acts.append(hres)
        return a_acts, h_acts

    def retain_grads(self):
        """Between forward and backward pass, ensure that relevant layers retain gradients for introspection."""
        for layer in self.enc.layers:
            if isinstance(layer, ISAB):
                layer.ab0.at.a.retain_grad()
                layer.ab0.hres.retain_grad()
                layer.ab1.at.a.retain_grad()
                layer.ab1.hres.retain_grad()

    def grads(self):
        """Get some relevant gradients for introspection (of the activations)"""
        a_grads = []
        h_grads = []
        for layer in self.enc.layers:
            if isinstance(layer, ISAB):
                if layer.ab0.at.a.grad is None:
                    return None  # no backward pass data

                a = torch.cat((layer.ab0.at.a.grad.flatten(), layer.ab1.at.a.grad.flatten()))
                a_grads.append(a)
                hres = torch.cat((layer.ab0.hres.grad.flatten(), layer.ab1.hres.grad.flatten()))
                h_grads.append(hres)
        return a_grads, h_grads

    def save(self, modelfile):
        torch.save({
            "modelState": self.state_dict(),
            "featureDims": self.featureDims,
            "depth": self.depth,
            "hiddenDims": self.hiddenDims,
            "queryDims": self.queryDims,
            "inducingPoints": self.inducingPoints
        }, modelfile)

    @staticmethod
    def load(modelfile):
        modeldata = torch.load(modelfile)
        featureDims = modeldata["featureDims"]
        depth = modeldata["depth"]
        hiddenDims = modeldata["hiddenDims"]
        queryDims = modeldata["queryDims"]
        inducingPoints = modeldata["inducingPoints"]
        model = StrengthNet(featureDims, depth, hiddenDims, queryDims, inducingPoints)
        model.load_state_dict(modeldata["modelState"])
        return model

class Sequential(nn.Module):
    """Like nn.Sequential, but passes xlens (collated minibatch structure) where necessary"""
    def __init__(self, *layers, introspection: bool = False):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.introspection = introspection
    
    def forward(self, x, xlens = None):
        hs = []
        for layer in self.layers:
            sig = inspect.signature(layer.forward)
            if 'xlens' in sig.parameters:
                x = layer(x, xlens)
            else:
                x = layer(x)
            if self.introspection:
                hs.append(x)
        self.hs = hs
        return x

class Attention(nn.Module):
    """Dot-product attention; no batching"""
    def __init__(self, queryDims: int, introspection: bool = False):
        super(Attention, self).__init__()
        self.z = torch.sqrt(torch.tensor(queryDims, dtype=torch.float32))
        self.introspection = introspection

    def forward(self, q, k, v):
        a = torch.matmul(q, k.transpose(-2, -1)) / self.z
        a = torch.softmax(a, dim=-1)
        if self.introspection:
            self.a = a
        h = torch.matmul(a, v)
        return h

class AttentionBlock(nn.Module):
    """Dot-product attention + FC + norms"""
    def __init__(self, hiddenDims: int, queryDims: int, introspection: bool = False):
        super(AttentionBlock, self).__init__()
        self.WQ = nn.Linear(hiddenDims, queryDims, bias=False)
        self.WK = nn.Linear(hiddenDims, queryDims, bias=False)
        self.WV = nn.Linear(hiddenDims, hiddenDims, bias=False)
        self.at = Attention(queryDims, introspection=introspection)
        self.fc = nn.Linear(hiddenDims, hiddenDims, bias=True)
        self.norm0 = nn.LayerNorm(hiddenDims)
        self.norm1 = nn.LayerNorm(hiddenDims)
        self.introspection = introspection

    def forward(self, q, qlens, h, hlens):
        qq = self.WQ(q)
        k, v = self.WK(h), self.WV(h)

        qslices = list(unbatch(q, qlens))
        hslices = list(unbatch(h, hlens))
        if not qlens:  # duplicate one query set over all h slices
            qslices = qslices * len(hslices)

        hs = []  # attention outputs

        for ((qstart, qend), (hstart, hend)) in zip(qslices, hslices):
            h = self.at(qq[qstart:qend], k[hstart:hend], v[hstart:hend])
            hs.append(h)

        h = torch.cat(hs, dim=0)
        h = h.view(-1, len(q), h.shape[-1])  # split batch dimension for broadcast add
        q = q.unsqueeze(0)  # add batch dimension of 1 for broadcast add
        h = self.norm0(q + h)
        h = h.flatten(0, 1)  # remove batch dimension

        hres = self.fc(h)
        if self.introspection:
            self.hres = hres
        h = self.norm1(h + torch.relu(hres))

        return h

class ISAB(nn.Module):
    """Induced set attention block"""
    def __init__(self, hiddenDims: int, queryDims: int, inducingPoints: int, introspection: bool = False):
        super(ISAB, self).__init__()
        self.i = nn.Parameter(torch.Tensor(inducingPoints, hiddenDims))
        nn.init.uniform_(self.i)
        self.ab0 = AttentionBlock(hiddenDims, queryDims, introspection=introspection)
        self.ab1 = AttentionBlock(hiddenDims, queryDims, introspection=introspection)

    def forward(self, x, xlens = None):
        h = self.ab0(self.i, None, x, xlens)  # (batchSize * inducingPoints) x hiddenDims
        hlens = [len(self.i)] * (len(h) // len(self.i))
        y = self.ab1(x, xlens, h, hlens)
        return y

class Pool(nn.Module):
    """Attention pooling layer with learned seed vector"""
    def __init__(self, hiddenDims: int, queryDims: int, introspection: bool = False):
        super(Pool, self).__init__()
        self.s = nn.Parameter(torch.Tensor(1, hiddenDims))
        nn.init.uniform_(self.s)
        self.ab = AttentionBlock(hiddenDims, queryDims, introspection=introspection)

    def forward(self, x, xlens = None):
        y = self.ab(self.s, None, x, xlens)
        return y

def unbatch(x, xlens):
    """Get an iterable over the batch elements in x"""
    # xlens specifies length of manually packed sequences in the batch
    if xlens:
        clens = np.cumsum([0] + xlens)
    else:
        clens = [0, len(x)]  # assume one input (eg in evaluation)
    return zip(clens[:-1], clens[1:])
