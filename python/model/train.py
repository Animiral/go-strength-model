import argparse
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from moves_dataset import MovesDataset, MovesDataLoader
from torch.nn.utils.rnn import pack_padded_sequence

import pdb

device = "cuda"

class StrengthNet(nn.Module):
    def __init__(self):
        d_hidden = 32
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(MovesDataset.featureDims, d_hidden),
            nn.ReLU()
        )
        self.rating = nn.Linear(d_hidden, 1)
        self.weights = nn.Linear(d_hidden, 1)
        self.softmax = nn.Softmax(dim=0)
        self.SCALE = 400 / math.log(10)  # Scale outputs to Elo/Glicko-like numbers

    def forward(self, x, xlens):
        # xlens specifies length of manually packed sequences in the batch
        clens = np.cumsum([0] + xlens)
        h = self.layer1(x)
        r = self.rating(h).squeeze(-1)
        z = self.weights(h).squeeze(-1)

        # predict one rating for each part in the batch
        parts = zip(clens[:-1], clens[1:])
        preds = [self._sumBySoftmax(r, z, start, end) for start, end in parts]
        return self.SCALE * torch.stack(preds)

    def _sumBySoftmax(self, r, z, start, end):
        if start == end:
            return torch.tensor(0, device=r.device)  # default prediction
        rslice = r[start:end]
        zslice = z[start:end]
        zslice = self.softmax(zslice)
        return torch.sum(zslice * rslice)

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    batch_size = args["batch_size"]
    steps = args["steps"]
    epochs = args["epochs"]

    print(f"Load training data from {listfile}")
    print(f"Batch size: {batch_size}")
    print(f"Steps: {steps}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")

    train_data = MovesDataset(listfile, featuredir, 'T')
    test_data = MovesDataset(listfile, featuredir, 'E')
    test_loader = MovesDataLoader(test_data, batch_size=batch_size)

    model = StrengthNet().to(device)
    print(f"Model parameters: {list(model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 5
    for e in range(epochs):
        sampler = BatchSampler(RandomSampler(train_data, replacement=True, num_samples=steps*batch_size), batch_size, False)
        train_loader = MovesDataLoader(train_data, batch_sampler=sampler)
        print(f"Epoch {e+1}\n-------------------------------")
        train(train_loader, model, optimizer, steps*batch_size)
        test(test_loader, model)
    print("Done!")

def train(loader, model, optimizer, totalsize: int=0):
    samples = 0  # how many we have learned
    model.train()
    MSE = nn.MSELoss()

    for batchnr, (bx, wx, blens, wlens, by, wy, score) in enumerate(loader):
        bx, by, wx, wy = map(lambda t: t.to(device), (bx, by, wx, wy))
        bpred, wpred = model(bx, blens), model(wx, wlens)
        loss = MSE(bpred, by) + MSE(wpred, wy) # + crossentropy(bt(bpred, wpred), score)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # status
        loss = loss.item()
        samples += len(score)
        print(f"loss: {loss:>7f}  [{samples:>5d}/{totalsize:>5d}]")

def test(loader, model):
    size = len(loader)
    model.eval()
    MSE = nn.MSELoss()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for bx, wx, blens, wlens, by, wy, score in loader:
            bx, by, wx, wy = map(lambda t: t.to(device), (bx, by, wx, wy))
            bpred, wpred = model(bx, blens), model(wx, wlens)
            loss = MSE(bpred, by) + MSE(wpred, wy)
            test_loss += loss.item()
    test_loss /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    description = """
    Train strength model on Go positions from dataset.
    """

    parser = argparse.ArgumentParser(description=description,add_help=False)
    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    required_args.add_argument('listfile', help='CSV file listing games and labels')
    required_args.add_argument('featuredir', help='directory containing extracted features')
    optional_args.add_argument('-b', '--batch-size', help='Minibatch size', type=int, default=100, required=False)
    optional_args.add_argument('-t', '--steps', help='Number of batches per epoch', type=int, default=100, required=False)
    optional_args.add_argument('-e', '--epochs', help='Nr of training epochs', type=int, default=5, required=False)

    args = vars(parser.parse_args())
    main(args)
