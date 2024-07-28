#!/usr/bin/env python3
# Our model training algorithm.

import argparse
import copy
import datetime
import math
from dataclasses import dataclass
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from moves_dataset import MovesDataset, MovesDataLoader, bradley_terry_score, scale_rating
from strengthnet import StrengthNet
from plots import trainingprogress
from plots import estimate_vs_label
from plots import netvis

device = "cuda"

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    featurename = args["featurename"]
    lowmemory = args["lowmemory"]
    animation = args["animation"]
    outfile = args["outfile"]
    trainlossfile = args["trainlossfile"]
    validationlossfile = args["validationlossfile"]
    figdir = args["figdir"]

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

    if animation:
        plt.ion()
        fig, axs = plt.subplots(2, 5, figsize=(20, 9.6))

        emb_lines = netvis.setup_embeddings(axs[1, 0], model)
        act_alines, act_hlines = netvis.setup_activations(axs[1, 1], axs[1, 2], model)
        grad_alines, grad_hlines = netvis.setup_gradients(axs[1, 3], axs[1, 4], model)
        figlines = (emb_lines, act_alines, act_hlines, grad_alines, grad_hlines)
    else:
        fig, axs = None, None

    def callback(model, epoch, trainlosses, validationlosses, record_v):
        log_progress(validationlossfile, trainlossfile, epoch, trainlosses, validationlosses)
        save_model(model, outfile, epoch)
        display(fig, axs, figlines, model, epoch, trainlosses, validationlosses, record_v, figdir)

    tparams = TrainingParams(
        epochs=epochs,
        steps=steps,
        batchSize=batchSize,
        learningrate=learningrate,
        lrdecay=lrdecay,
        patience=patience,
        tauRatings=tauRatings,
        tauL2=tauL2)
    training = Training(model, trainData, validationLoader, tparams, animation)
    bestmodel, bestloss, trainlosses, validationlosses = training.run(callback)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\t[{timestamp}] Training done, best validation loss: {bestloss}")

    save_model(bestmodel, outfile)

    trainlossfile and trainlossfile.close()
    validationlossfile and validationlossfile.close()

    if animation:
        plt.ioff()
        plt.show()  # block to show final charts

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
        l_score = -(1 - (score - bradley_terry_score(bpred, wpred)).abs_()).log_().mean()
        l_ratings = self.mse(bpred, by) + self.mse(wpred, wy)
        l_l2 = sum(p.pow(2).sum() for p in self.model.parameters()) / self.parametersCount
        return l_score, self.tauRatings * l_ratings, self.tauL2 * l_l2

    def validationLoss(self, bpred, wpred, by, wy, score):
        l_score = -(1 - (score - bradley_terry_score(bpred, wpred)).abs_()).log_().mean()
        l_ratings = self.mse(bpred, by) + self.mse(wpred, wy)
        return l_score, self.tauRatings * l_ratings

    def scoreSamples(self, bpred, wpred, score):
        """Return the score estimate from the model's outputs, partitioned by actual winner"""
        spred = bradley_terry_score(bpred, wpred)
        spred_white = spred[score < 0.5]  # score predictions when white is the actual winner
        spred_black = spred[score > 0.5]  # score predictions when black is the actual winner
        return spred_white, spred_black

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
    """Implements the training components, which can be configured with different hyperparameters."""

    def __init__(self, model, trainData: MovesDataset, validationLoader: MovesDataLoader, tparams: TrainingParams, animation: bool = False):
        # general data and parameters
        self.trainData = trainData
        self.validationLoader = validationLoader
        self.tparams = tparams
        self.animation = animation  # if this is set, collect more data to display

        # training state
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.tparams.learningrate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=1/self.tparams.lrdecay)
        self.loss = StrengthNetLoss(model, self.tparams.tauRatings, self.tparams.tauL2)

    def run(self, callback):
        """Train the model for a number of epochs with early stopping and update callbacks"""
        model = self.model
        scheduler = self.scheduler

        bestmodel = copy.deepcopy(model)
        bestloss = float("inf")
        currentEpoch = 0
        badEpochs = 0  # early stopping counter until patience runs out
        trainlosses = []
        validationlosses = []

        l_vs, l_vr, record_v = self.validate()
        validationlosses.append((l_vs, l_vr))
        # validationlosses.append((1, 1)) # skip initial validation for faster debugging
        callback(model, 0, trainlosses, validationlosses, record_v)

        for e in range(self.tparams.epochs):
            trainlosses += self.epoch()
            l_vs, l_vr, record_v = self.validate()
            validationlosses.append((l_vs, l_vr))
            callback(model, e+1, trainlosses, validationlosses, record_v)

            # stop early?
            if l_vs < bestloss:  # performance = score loss. we only have ratings loss for reference.
                bestmodel = copy.deepcopy(model)
                bestloss = l_vs
                badEpochs = 0
            else:
                badEpochs += 1
                if badEpochs >= self.tparams.patience:
                    break

            scheduler.step()  # decay learning rate

        return bestmodel, bestloss, trainlosses, validationlosses

    def epoch(self, model = None, data = None, steps = 0, batchSize = 0, optimizer = None, loss = None):
        """Train the model for one epoch"""
        model = model or self.model
        data = data or self.trainData
        steps = steps or self.tparams.steps
        batchSize = batchSize or self.tparams.batchSize
        optimizer = optimizer or self.optimizer
        loss = loss or self.loss

        num_samples = steps*batchSize
        rndsampler = RandomSampler(data, replacement=True, num_samples=num_samples)
        sampler = BatchSampler(rndsampler, batchSize, False)
        loader = MovesDataLoader(data, batch_sampler=sampler)

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
            l_score = l_score.item()
            l_ratings = l_ratings.item()
            l_l2 = l_l2.item()
            trainloss.append((l_score, l_ratings, l_l2))

        return trainloss

    def validate(self, loader = None, model = None, loss = None):
        """Validation test the model"""
        loader = loader or self.validationLoader
        model = model or self.model
        loss = loss or self.loss
        animation = self.animation

        batches = len(loader)
        model.eval()

        # all results
        preds = []  # rank estimates by model
        ys = []     # validation set rank labels
        spreds_white = []  # score estimate when white wins
        spreds_black = []  # score estimate when black wins
        l_score, l_ratings = 0, 0

        with torch.no_grad():
            for bx, wx, blens, wlens, by, wy, score in loader:
                bx, by, wx, wy, score = map(lambda t: t.to(device), (bx, by, wx, wy, score))
                bpred, wpred = model(bx, blens), model(wx, wlens)
                l_s, l_r = loss.validationLoss(bpred, wpred, by, wy, score)
                l_score += l_s.item()
                l_ratings += l_r.item()
                if animation:
                    preds.append(bpred.cpu().numpy())
                    preds.append(wpred.cpu().numpy())
                    ys.append(wy.cpu().numpy())
                    ys.append(by.cpu().numpy())
                    spred_white, spred_black = loss.scoreSamples(bpred, wpred, score)
                    spreds_white.append(spred_white.cpu().numpy())
                    spreds_black.append(spred_black.cpu().numpy())

        # if animation:
        #     # get model gradients
        #     model.retain_grads()
        #     l = l_s + l_r
        #     l.backward()

        l_score /= batches
        l_ratings /= batches
        preds, ys, spreds_white, spreds_black = map(np.concatenate, (preds, ys, spreds_white, spreds_black))
        return l_score, l_ratings, (preds, ys, spreds_white, spreds_black)

def log_progress(validationlossfile, trainlossfile, epoch, trainlosses, validationlosses):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if trainlosses:
        lt_score, lt_ratings, lt_l2 = trainlosses[-1]
        lt = lt_score + lt_ratings + lt_l2
    else:
        lt = lt_score = lt_ratings = lt_l2 = float("inf")
    lv_score, lv_ratings = validationlosses[-1]
    print(f"[{timestamp}] Epoch {epoch} error: training {lt_score:>8f}(s) + {lt_ratings:>8f}(r) + {lt_l2:>8f}(L2) = {lt:>8f}, validation {lv_score:>8f}(s), {lv_ratings:>8f}(r)")

    if validationlossfile:
        validationlossfile.write(f"{lv_score},{lv_ratings}\n")
    if trainlossfile and trainlosses:
        for (lt_score, lt_ratings, lt_l2) in trainlosses:
            trainlossfile.write(f"{lt_score},{lt_ratings},{lt_l2}\n")

def save_model(model, outfile, epoch=None):
    if outfile:
        epochstr = "" if epoch is None else str(epoch+1) 
        modelfile = outfile.replace("{}", epochstr)
        model.save(modelfile)

def display(fig, axs, figlines, model, epoch, trainlosses, validationlosses, record_v, figdir):
    if fig is None:
        return

    # training progress
    axs[0, 0].clear()
    trainingprogress.setup(axs[0, 0])
    trainingprogress.plot(axs[0, 0], trainlosses, validationlosses)

    # model vs labels
    preds, ys, spreds_white, spreds_black = record_v
    preds = scale_rating(preds)
    ys = scale_rating(ys)
    axs[0, 1].clear()
    estimate_vs_label.setup_ratings(axs[0, 1], f"Epoch {epoch}", "Validation Set")
    estimate_vs_label.plot_ratings(axs[0, 1], ys, preds)
    axs[0, 2].clear()
    axs[0, 3].clear()
    estimate_vs_label.setup_score(axs[0, 2], axs[0, 3])
    estimate_vs_label.plot_score(axs[0, 2], axs[0, 3], np.sort(spreds_white), np.sort(spreds_black))

    # net visualization
    emb_lines, act_alines, act_hlines, grad_alines, grad_hlines = figlines
    netvis.plot_embeddings(axs[1, 0], emb_lines, model)
    netvis.plot_activations(axs[1, 1], axs[1, 2], act_alines, act_hlines, model)
    netvis.plot_gradients(axs[1, 3], axs[1, 4], grad_alines, grad_hlines, model)

    # This is required for the current plot to actually draw using Qt backend
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    if figdir:
        with open(f"{figdir}/epoch{epoch}.pkl", "wb") as f:
            pickle.dump(fig, f)                         # for later re-loading if necessary
        fig.savefig(f"{figdir}/epoch{epoch}.pdf")       # for offline viewing

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
    optional_args.add_argument("--figdir", help="Output directory to store figures that visualize training", type=str, required=False)
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
