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
    lowmemory = args["lowmemory"]
    outfile = args["outfile"]
    trainlossfile = args["trainlossfile"]
    validationlossfile = args["validationlossfile"]

    batchSize = args["batch_size"]
    steps = args["steps"]
    epochs = args["epochs"]
    learningrate = args["learningrate"]
    lrdecay = args["lrdecay"]
    patience = args["patience"]
    tauRatings = args["tau_ratings"]
    tauL2 = args["tau_l2"]

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

    trainData = MovesDataset(listfile, featuredir, "T", featurename=featurename, featurememory=not lowmemory)
    validationData = MovesDataset(listfile, featuredir, "V", featurename=featurename, featurememory=not lowmemory)
    validationLoader = MovesDataLoader(validationData, batch_size=batchSize)
    model = StrengthNet(trainData.featureDims, depth, hiddenDims, queryDims, inducingPoints)
    model = model.to(device)

    def callback(model, e, trainloss, validationloss):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if trainloss:
            lt_score, lt_ratings, lt_l2 = trainloss[-1]
            lt = lt_score + lt_ratings + lt_l2
        else:
            lt = lt_score = lt_ratings = lt_l2 = float("inf")
        lv_score, lv_ratings = validationloss
        print(f"[{timestamp}] Epoch {e} error: training {lt_score:>8f}(s) + {lt_ratings:>8f}(r) + {lt_l2:>8f}(L2) = {lt:>8f}, validation {lv_score:>8f}(s), {lv_ratings:>8f}(r)")

        if validationlossfile:
            validationlossfile.write(f"{lv_score},{lv_ratings}\n")
        if trainlossfile and trainloss:
            for (lt_score, lt_ratings, lt_l2) in trainloss:
                trainlossfile.write(f"{lt_score},{lt_ratings},{lt_l2}\n")
        if outfile:
            modelfile = outfile.replace("{}", str(e+1))
            model.save(modelfile)

    tparams = TrainingParams(epochs=epochs, steps=steps, batchSize=batchSize,
        learningrate=learningrate, lrdecay=lrdecay, patience=patience,
        tauRatings=tauRatings, tauL2=tauL2)
    t = Training(callback, trainData, validationLoader, tparams)
    bestmodel, validationloss = t.run(model)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\t[{timestamp}] Training done, best validation loss: {validationloss}")

    if outfile:
        modelfile = outfile.replace("{}", "")
        model.save(modelfile)

    trainlossfile and trainlossfile.close()
    validationlossfile and validationlossfile.close()

class StrengthNetLoss:

    # being off by 500 Glicko-2 points in both bpred and wpred is
    # as bad as getting the score half wrong.
    DEFAULT_TAU_RATINGS = -math.log(.5) / (2*(500/MovesDataset.GLICKO2_STDEV)**2)
    # all parameter == 1 is as bad as getting the score half wrong.
    DEFAULT_TAU_L2 = -math.log(.5)

    def __init__(self, model, tauRatings=None, tauL2=None):
        if tauRatings is None:
            tauRatings = StrengthNetLoss.DEFAULT_TAU_RATINGS
        if tauL2 is None:
            tauL2 = StrengthNetLoss.DEFAULT_TAU_L2

        self.model = model
        self.parametersCount = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.tauRatings = tauRatings
        self.tauL2 = tauL2
        self.mse = nn.MSELoss()

    def trainLoss(self, bpred, wpred, by, wy, score):
        l_score = -(1 - (score - bradley_terry_score(bpred, wpred)).abs_()).log_().sum()
        l_ratings = self.mse(bpred, by) + self.mse(wpred, wy)
        l_l2 = sum(p.pow(2).sum() for p in self.model.parameters()) / self.parametersCount
        return l_score, self.tauRatings * l_ratings, self.tauL2 * l_l2

    def validationLoss(self, bpred, wpred, by, wy, score):
        l_score = -(1 - (score - bradley_terry_score(bpred, wpred)).abs_()).log_().sum()
        l_ratings = self.mse(bpred, by) + self.mse(wpred, wy)
        return l_score, self.tauRatings * l_ratings

@dataclass
class TrainingParams:
    epochs: int
    steps: int
    batchSize: int
    learningrate: float
    lrdecay: float
    patience: int
    tauRatings: float
    tauL2: float

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
        loss = StrengthNetLoss(model, self.tparams.tauRatings, self.tparams.tauL2)
        bestmodel = copy.deepcopy(model)
        bestloss = float("inf")
        badEpochs = 0
        validationloss = self.validate(self.validationLoader, model, loss)
        self.callback(model, 0, [], validationloss)

        for e in range(self.tparams.epochs):
            trainloss = self.epoch(model, optimizer, loss)
            validationloss = self.validate(self.validationLoader, model, loss)
            epochloss, _ = validationloss  # performance = score loss. we only have ratings loss for reference.
            self.callback(model, e+1, trainloss, validationloss)

            if epochloss < bestloss:
                bestmodel = copy.deepcopy(model)
                bestloss = epochloss
                badEpochs = 0
            else:
                badEpochs += 1
                if badEpochs >= self.tparams.patience:
                    break

            scheduler.step()  # decay learning rate

        return bestmodel, bestloss

    def epoch(self, model, optimizer, loss):
        num_samples = self.tparams.steps*self.tparams.batchSize
        rndsampler = RandomSampler(self.trainData, replacement=True, num_samples=num_samples)
        sampler = BatchSampler(rndsampler, self.tparams.batchSize, False)
        loader = MovesDataLoader(self.trainData, batch_sampler=sampler)

        model.train()
        trainloss = []

        for batchnr, (bx, wx, blens, wlens, by, wy, score) in enumerate(loader):
            bx, by, wx, wy, score = map(lambda t: t.to(device), (bx, by, wx, wy, score))
            bpred, wpred = model(bx, blens), model(wx, wlens)
            l_score, l_ratings, l_l2 = loss.trainLoss(bpred, wpred, by, wy, score)
            l = l_score + l_ratings + l_l2
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            # keep track of loss
            batchSize = len(score)
            l_score = l_score.item() / batchSize
            l_ratings = l_ratings.item() / batchSize
            l_l2 = l_l2.item() / batchSize
            trainloss.append((l_score, l_ratings, l_l2))

        return trainloss

    def validate(self, loader, model, loss):
        batches = len(loader)
        model.eval()

        l_score, l_ratings = 0, 0
        with torch.no_grad():
            for bx, wx, blens, wlens, by, wy, score in loader:
                bx, by, wx, wy, score = map(lambda t: t.to(device), (bx, by, wx, wy, score))
                bpred, wpred = model(bx, blens), model(wx, wlens)
                l_s, l_r = loss.validationLoss(bpred, wpred, by, wy, score)
                batchSize = len(score)
                l_score += l_s.item() / batchSize
                l_ratings += l_r.item() / batchSize
        l_score /= batches
        l_ratings /= batches
        return l_score, l_ratings

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
    optional_args.add_argument("-m", "--lowmemory", help="Do not keep entire dataset in memory", action="store_true", required=False)
    optional_args.add_argument("-a", "--animation", help="Visualize the training process using continuously updating plots", action="store_true", required=False)
    optional_args.add_argument("-o", "--outfile", help="Pattern for model output, with epoch placeholder \"{}\" ", type=str, required=False)
    optional_args.add_argument("--trainlossfile", help="Output file to store training loss values", type=str, required=False)
    optional_args.add_argument("--validationlossfile", help="Output file to store validation loss values", type=str, required=False)
    optional_args.add_argument("-b", "--batch-size", help="Minibatch size", type=int, default=100, required=False)
    optional_args.add_argument("-t", "--steps", help="Number of batches per epoch", type=int, default=100, required=False)
    optional_args.add_argument("-e", "--epochs", help="Nr of training epochs", type=int, default=5, required=False)
    optional_args.add_argument("-l", "--learningrate", help="Initial gradient scale", type=float, default=1e-3, required=False)
    optional_args.add_argument("-d", "--lrdecay", help="Leraning rate decay", type=float, default=0.95, required=False)
    optional_args.add_argument("-p", "--patience", help="Epochs without improvement before early stop", type=int, default=3, required=False)
    optional_args.add_argument("--tau-ratings", help="Scaling factor for rating labels MSE", type=float, default=None, required=False)
    optional_args.add_argument("--tau-l2", help="Scaling factor for parameter regularization", type=float, default=None, required=False)
    optional_args.add_argument("-D", "--depth", help="Number of consecutive attention blocks in model", type=int, default=1, required=False)
    optional_args.add_argument("-H", "--hidden-dims", help="Hidden feature dimensionality", type=int, default=8, required=False)
    optional_args.add_argument("-Q", "--query-dims", help="Query feature dimensionality", type=int, default=8, required=False)
    optional_args.add_argument("-I", "--inducing-points", help="Number of inducing points in attention mechanism", type=int, default=1, required=False)

    args = vars(parser.parse_args())
    main(args)
