#!/usr/bin/env python3
# Our model training algorithm.

import argparse
import copy
import datetime
import math
from dataclasses import dataclass
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
    patience = args["patience"]

    windowSize = args["window_size"]
    depth = args["depth"]
    hiddenDims = args["hidden_dims"]
    queryDims = args["query_dims"]
    inducingPoints = args["inducing_points"]

    for k, v in args.items():
        print(f"{k}: {v}")
    print(f"Device: {device}")

    if trainlossfile:
        trainlossfile = open(trainlossfile, "w")
    if validationlossfile:
        validationlossfile = open(validationlossfile, "w")

    trainData = MovesDataset(listfile, featuredir, "T", featurename=featurename)
    validationData = MovesDataset(listfile, featuredir, "V", featurename=featurename)
    validationLoader = MovesDataLoader(windowSize, validationData, batch_size=batchSize)
    model = StrengthNet(trainData.featureDims, depth, hiddenDims, queryDims, inducingPoints)
    model = model.to(device)

    def callback(model, e, trainloss, validationloss):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Epoch {e} error: training {trainloss[-1]:>8f}, validation {validationloss:>8f}")
        if validationlossfile:
            validationlossfile.write(f"{validationloss}\n")
        if trainlossfile:
            for loss in trainloss:
                trainlossfile.write(f"{loss}\n")
        if outfile:
            modelfile = outfile.replace("{}", str(e+1))
            model.save(modelfile)

    tparams = TrainingParams(epochs=epochs, steps=steps, batchSize=batchSize, learningrate=learningrate, lrdecay=lrdecay, windowSize=windowSize, patience=patience)
    t = Training(callback, trainData, validationLoader, tparams)
    bestmodel, validationloss = t.run(model)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\t[{timestamp}] Training done, best validation loss: {validationloss}")

    if outfile:
        modelfile = outfile.replace("{}", "")
        model.save(modelfile)

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

@dataclass
class TrainingParams:
    epochs: int
    steps: int
    batchSize: int
    learningrate: float
    lrdecay: float
    windowSize: int
    patience: int

class Training:
    """Implements the training loop, which can be run with different hyperparameters."""

    def __init__(self, callback, trainData: MovesDataset, validationLoader: MovesDataLoader, tparams: TrainingParams):
        self.callback = callback
        self.trainData = trainData
        self.validationLoader = validationLoader
        self.tparams = tparams
        self.badEpochs = 0  # early stopping counter until patience runs out

    def run(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.tparams.learningrate)
        scheduler = StepLR(optimizer, step_size=1, gamma=1/self.tparams.lrdecay)
        bestmodel = copy.deepcopy(model)
        bestloss = float("inf")
        badEpochs = 0
        validationloss = self.validate(self.validationLoader, model)
        self.callback(model, 0, [float("inf")], validationloss)

        for e in range(self.tparams.epochs):
            trainloss = self.epoch(model, optimizer)
            validationloss = self.validate(self.validationLoader, model)
            self.callback(model, e+1, trainloss, validationloss)

            if validationloss < bestloss:
                bestmodel = copy.deepcopy(model)
                bestloss = validationloss
                badEpochs = 0
            else:
                badEpochs += 1
                if badEpochs >= self.tparams.patience:
                    break

            scheduler.step()  # decay learning rate

        return bestmodel, bestloss

    def epoch(self, model, optimizer):
        num_samples = self.tparams.steps*self.tparams.batchSize
        rndsampler = RandomSampler(self.trainData, replacement=True, num_samples=num_samples)
        sampler = BatchSampler(rndsampler, self.tparams.batchSize, False)
        loader = MovesDataLoader(self.tparams.windowSize, self.trainData, batch_sampler=sampler)

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
                batchSize = len(score)
                test_loss += l.item() / batchSize
        test_loss /= batches
        return test_loss

if __name__ == "__main__":
    description = """
    Train strength model on Go positions from dataset.
    """

    parser = argparse.ArgumentParser(description=description,add_help=False)
    required_args = parser.add_argument_group("required arguments")
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit"
    )
    required_args.add_argument("listfile", help="CSV file listing games and labels")
    required_args.add_argument("featuredir", help="Directory containing extracted features")
    optional_args.add_argument("-f", "--featurename", help="Type of features to train on", type=str, default="pick", required=False)
    optional_args.add_argument("-o", "--outfile", help="Pattern for model output, with epoch placeholder \"{}\" ", type=str, required=False)
    optional_args.add_argument("--trainlossfile", help="Output file to store training loss values", type=str, required=False)
    optional_args.add_argument("--validationlossfile", help="Output file to store validation loss values", type=str, required=False)
    optional_args.add_argument("-b", "--batch-size", help="Minibatch size", type=int, default=100, required=False)
    optional_args.add_argument("-t", "--steps", help="Number of batches per epoch", type=int, default=100, required=False)
    optional_args.add_argument("-e", "--epochs", help="Nr of training epochs", type=int, default=5, required=False)
    optional_args.add_argument("-l", "--learningrate", help="Initial gradient scale", type=float, default=1e-3, required=False)
    optional_args.add_argument("-d", "--lrdecay", help="Leraning rate decay", type=float, default=0.95, required=False)
    optional_args.add_argument("-p", "--patience", help="Epochs without improvement before early stop", type=int, default=3, required=False)
    optional_args.add_argument("-w", "--window-size", help="Maximum number of recent moves", type=int, default=500, required=False)
    optional_args.add_argument("-D", "--depth", help="Number of consecutive attention blocks in model", type=int, default=1, required=False)
    optional_args.add_argument("-H", "--hidden-dims", help="Hidden feature dimensionality", type=int, default=8, required=False)
    optional_args.add_argument("-Q", "--query-dims", help="Query feature dimensionality", type=int, default=8, required=False)
    optional_args.add_argument("-I", "--inducing-points", help="Number of inducing points in attention mechanism", type=int, default=1, required=False)

    args = vars(parser.parse_args())
    main(args)
