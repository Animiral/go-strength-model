#!/usr/bin/env python3
# Our model training algorithm.

import argparse
import os
import copy
import math
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from moves_dataset import MovesDataset, MovesDataLoader, bradley_terry_score
from strengthnet import StrengthNet
from torch.optim.lr_scheduler import StepLR

device = "cuda"

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    featurename = args["featurename"]
    outfile = args["outfile"]
    trainlossfile = args["trainlossfile"]
    validationlossfile = args["validationlossfile"]
    batchSize = args["batch_size"]
    steps = args["steps"]
    epochs = args["epochs"]
    learningrate = args["learningrate"]
    lrdecay = args["lrdecay"]

    print(f"Load training data from {listfile}")
    print(f"Load precomputed {featurename} features from {featuredir}")
    print(f"Save model(s) to {outfile}")
    print(f"Batch size: {batch_size}")
    print(f"Steps: {steps}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")

    if trainlossfile:
        print(f"Write training loss to {trainlossfile}")
        trainlossfile = open(trainlossfile, "w")
    if validationlossfile:
        print(f"Write validation loss to {validationlossfile}")
        validationlossfile = open(validationlossfile, "w")

    trainData = MovesDataset(listfile, featuredir, "T", featurename=featurename)
    validationData = MovesDataset(listfile, featuredir, "V", featurename=featurename)

    windowSize = 500
    depth = args.get("modeldepth", 2)
    hiddenDims = args.get("hidden_dims", 16)
    queryDims = args.get("query_dims", 8)
    inducingPoints = args.get("inducing_points", 8)

    validationLoader = MovesDataLoader(windowSize, validationData, batch_size=batchSize)
    model = StrengthNet(featureDims, depth, hiddenDims, queryDims, inducingPoints)
    model = model.to(device)

    def callback(model, e, trainloss, validationloss):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\t[{timestamp}] Epoch {e} error: training {trainloss[-1]:>8f}, validation {validationloss:>8f}")
        if validationlossfile:
            validationlossfile.write(f"{validationloss}\n")
        if trainlossfile:
            for loss in trainloss:
                trainlossfile.write(f"{loss}\n")
        if outfile:
            modelfile = outfile.replace('{}', str(e+1))
            torch.save({
                "modelState": model.state_dict(),
                "featureDims": model.featureDims,
                "depth": model.depth,
                "hiddenDims": model.hiddenDims,
                "queryDims": model.queryDims,
                "inducingPoints": model.inducingPoints
            }, modelfile)

    t = Training(callback, trainData, validationLoader,
                 epochs, steps, batchSize, learningrate, lrdecay, windowSize)
    bestmodel, validationloss = t.run(model)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\t[{timestamp}] Training done, best validation loss: {validationloss}")

    trainlossfile and trainlossfile.close()
    validationlossfile and validationlossfile.close()

def loss(bpred, wpred, by, wy, score, tau=None):
    if tau is None:
        # default: being off by 200 Glicko-2 points in both bpred and wpred is
        # as bad as getting the score half wrong.
        tau = 2*(200/MovesDataset.GLICKO2_STDEV)**2 / -math.log(.5)
    MSE = nn.MSELoss()
    rating_loss = MSE(bpred, by) + MSE(wpred, wy)
    score_loss = -(1 - (score - bradley_terry_score(bpred, wpred)).abs_()).log_().sum()
    return rating_loss + tau * score_loss

class Training:
    """Implements the training loop, which can be run with different hyperparameters."""

    def __init__(self, callback, trainData: MovesDataset, validationLoader: MovesDataLoader,
                 epochs: int, steps: int, batchSize: int,
                 learningrate: float, lrdecay: float, windowSize: int):
        self.callback = callback
        self.trainData = trainData
        self.validationLoader = validationLoader
        self.epochs = epochs
        self.steps = steps
        self.batchSize = batchSize
        self.learningrate = learningrate
        self.lrdecay = lrdecay
        self.patience = 3
        self.badEpochs = 0  # early stopping counter until patience runs out
        self.windowSize = windowSize

    def run(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learningrate)
        scheduler = StepLR(optimizer, step_size=1, gamma=1/self.lrdecay)
        bestmodel = copy.deepcopy(model)
        bestloss = float("inf")
        badEpochs = 0
        validationloss = self.validate(self.validationLoader, model)
        self.callback(model, 0, [float("inf")], validationloss)

        for e in range(self.epochs):
            trainloss = self.epoch(model, optimizer)
            validationloss = self.validate(self.validationLoader, model)
            self.callback(model, e+1, trainloss, validationloss)

            if validationloss < bestloss:
                bestmodel = copy.deepcopy(model)
                bestloss = validationloss
                badEpochs = 0
            else:
                badEpochs += 1
                if badEpochs >= self.patience:
                    break

            scheduler.step()  # decay learning rate

        return bestmodel, bestloss

    def epoch(self, model, optimizer):
        sampler = BatchSampler(RandomSampler(self.trainData, replacement=True, num_samples=self.steps*self.batchSize), self.batchSize, False)
        loader = MovesDataLoader(self.windowSize, self.trainData, batch_sampler=sampler)

        model.train()
        trainloss = []

        for batchnr, (bx, wx, blens, wlens, by, wy, score) in enumerate(loader):
            bx, by, wx, wy, score = map(lambda t: t.to(device), (bx, by, wx, wy, score))
            bpred, wpred = model(bx, blens), model(wx, wlens)
            l = loss(bpred, wpred, by, wy, score)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            # keep track of loss
            batchSize = len(score)
            l = l.item() / batchSize
            trainloss.append(l)

        return trainloss

    def validate(self, loader, model):
        batches = len(loader)
        model.eval()

        test_loss, correct = 0, 0
        with torch.no_grad():
            for bx, wx, blens, wlens, by, wy, score in loader:
                bx, by, wx, wy, score = map(lambda t: t.to(device), (bx, by, wx, wy, score))
                bpred, wpred = model(bx, blens), model(wx, wlens)
                l = loss(bpred, wpred, by, wy, score)
                test_loss += l.item()
        test_loss /= batches
        return test_loss

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
    required_args.add_argument('featuredir', help='Directory containing extracted features')
    optional_args.add_argument('-f', '--featurename', help='Type of features to train on', type=str, default='pick', required=False)
    optional_args.add_argument('-o', '--outfile', help='Pattern for model output, with epoch placeholder "{}" ', type=str, required=False)
    optional_args.add_argument('-b', '--batch-size', help='Minibatch size', type=int, default=100, required=False)
    optional_args.add_argument('-t', '--steps', help='Number of batches per epoch', type=int, default=100, required=False)
    optional_args.add_argument('-e', '--epochs', help='Nr of training epochs', type=int, default=5, required=False)
    optional_args.add_argument('-l', '--learningrate', help='Initial gradient scale', type=float, default=1e-3, required=False)
    optional_args.add_argument('-d', '--lrdecay', help='Leraning rate decay', type=float, default=0.95, required=False)
    optional_args.add_argument('--trainlossfile', help='Output file to store training loss values', type=str, required=False)
    optional_args.add_argument('--validationlossfile', help='Output file to store validation loss values', type=str, required=False)

    args = vars(parser.parse_args())
    main(args)
