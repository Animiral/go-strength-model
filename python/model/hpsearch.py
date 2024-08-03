#!/usr/bin/env python3
# Hyperparameter search algorithm.

import argparse
import os
import sys
import copy
import datetime
import math
import re
import random
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
import subprocess
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from moves_dataset import MovesDataset, MovesDataLoader, bradley_terry_score
from train import Training, TrainingParams, StrengthNetLoss
from strengthnet import StrengthNet
from torch.optim.lr_scheduler import StepLR

device = "cuda"

@dataclass
class HyperParams:
    """Includes all training params that we optimize with hp search."""
    learningrate: float
    lrdecay: float
    tauRatings: float    # scale of labels MSE in training loss
    tauL2: float         # scale of param regularization in training loss
    depth: int           # model layers
    hiddenDims: int      # hidden feature dimensionality
    queryDims: int       # query feature dimensionality
    inducingPoints: int  # number of query vectors in ISAB

def main(args):
    title = args["title"]
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    featurename = args["featurename"]
    netdir = args["netdir"]
    logdir = args["logdir"]
    batchSize = args["batch_size"]
    steps = args["steps"]
    epochs = args["epochs"]
    workers = args["workers"]
    resume = args["resume"]
    patience = args["patience"]
    samples = args["samples"]
    iterations = args["iterations"]
    decay = args["decay"]

    for k, v in args.items():
        print(f"{k}: {v}")
    print(f"Device: {device}")

    logfile = open(f"{logdir}/{title}.txt", "w")

    def logMessage(message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")
        logfile.write(f"[{timestamp}] {message}\n")

    search = HyperparamSearch(title, listfile, featuredir, featurename, netdir, logdir,
                              workers, resume, epochs, steps, batchSize, patience, samples)
    scale = 1.0
    modelfile = None

    for i in range(iterations):
        logMessage(f"=== Hyperparameter Search: Iteration {i}, scale={scale} ===")
        hparams, modelfile, validationloss = search.search(scale)
        logMessage(f"Best hparams (vloss={validationloss}) it{i}: {hparams}")
        scale *= decay

    if modelfile:
        modelfile = os.path.abspath(modelfile)  # link to abs path is best practice
        bestfile = f"{netdir}/{title}/model.pth"
        os.symlink(modelfile, bestfile)

    logfile.close()

class HyperparamSearch:
    """Run multiple trainings with different hyperparameters."""

    def __init__(self, title: str, # trainData: MovesDataset, validationData: MovesDataset,
                 listfile: str, featuredir: str, featurename: str,
                 netdir: str, logdir: str, workers: int, resume: bool,
                 epochs: int, steps: int, batchSize: int,
                 patience: int, trainingSamples: int):
        self.title = title
        # self.trainData = trainData
        # self.validationData = validationData
        self.listfile = listfile
        self.featuredir = featuredir
        self.featurename = featurename
        os.makedirs(f"{netdir}/{title}", exist_ok=True)
        os.makedirs(f"{logdir}/{title}", exist_ok=True)
        self.iteration = 0
        self.netdir = netdir
        self.logdir = logdir
        self.workers = workers
        self.resume = resume
        self.epochs = epochs
        self.steps = steps
        self.batchSize = batchSize
        self.patience = patience
        self.trainingSamples = trainingSamples
        hparams = HyperParams(learningrate = 10**-3,
                              lrdecay = 0.98,
                              tauRatings = StrengthNetLoss.DEFAULT_TAU_RATINGS,
                              tauL2 = StrengthNetLoss.DEFAULT_TAU_L2,
                              depth = 3,
                              hiddenDims = 200,
                              queryDims = 40,
                              inducingPoints = 40)
        self.best = (hparams, None, float("inf"))

    def search(self, scale):
        hparams = [self.randomParams(self.best[0], scale) for _ in range(self.trainingSamples)]

        with ThreadPool(self.workers) as pool:
            samples = pool.starmap(self.training, enumerate(hparams))

        self.best = min(list(samples) + [self.best], key=lambda x: x[2])
        self.iteration += 1
        return self.best

    def searchBroad(self):
        return self.search(1)

    def searchFine(self):
        return self.search(0.1)

    def randomParams(self, hparams: HyperParams, scale: float):
        """Return `hparams` adjusted randomly proportional to `scale`"""
        hparams = copy.deepcopy(hparams)
        hparams.learningrate *= 10**random.uniform(-3*scale, 3*scale)
        hparams.lrdecay += random.uniform(-0.05*scale, 0.05*scale)
        hparams.tauRatings *= 2**random.uniform(-3*scale, 3*scale)
        hparams.tauL2 *= 2**random.uniform(-3*scale, 3*scale)
        hparams.depth = random.randint(1, 5)
        hparams.hiddenDims = int(math.ceil(hparams.hiddenDims * (2**random.uniform(-2*scale, 2*scale))))
        hparams.queryDims = int(math.ceil(hparams.queryDims * (2**random.uniform(-2*scale, 2*scale))))
        hparams.inducingPoints += random.randint(round(-32*scale), round(32*scale))
        return self.clampParams(hparams)

    def clampParams(self, hparams: HyperParams):
        hparams.learningrate = 10**-5 if hparams.learningrate < 10**-5 else 1 if hparams.learningrate > 1 else hparams.learningrate
        hparams.lrdecay = 0.9 if hparams.lrdecay < 0.9 else 1 if hparams.lrdecay > 1 else hparams.lrdecay
        hparams.tauRatings = 0.001 if hparams.tauRatings < 0.001 else 10.0 if hparams.tauRatings > 10.0 else hparams.tauRatings
        hparams.tauL2 = 0.001 if hparams.tauL2 < 0.001 else 100.0 if hparams.tauL2 > 100.0 else hparams.tauL2
        hparams.depth = 1 if hparams.depth < 1 else 5 if hparams.depth > 5 else hparams.depth
        hparams.hiddenDims = 8 if hparams.hiddenDims < 8 else 256 if hparams.hiddenDims > 256 else hparams.hiddenDims
        hparams.queryDims = 8 if hparams.queryDims < 8 else 256 if hparams.queryDims > 256 else hparams.queryDims
        hparams.inducingPoints = 1 if hparams.inducingPoints < 1 else 64 if hparams.inducingPoints > 64 else hparams.inducingPoints
        return hparams

    def loadResults(self, logpath: str):
        """
        Read the hparams and validationloss from a search
        training run that logged to the given file.
        """
        hpattern = re.compile(
            r"HP Search it(?P<iteration>\d+) seq(?P<sequence>\d+) \| HyperParams\((?P<hyperparameters>[^\)]+)\)"
        )
        vlpattern = re.compile(r"validation\s+([0-9.]+)")

        with open(logpath, "r") as logfile:
            lines = logfile.readlines()

        match = hpattern.search(lines[0])
        if not match:
            raise ValueError(f"Hyperparameters not found in log file {logpath}")

        iteration = match.group("iteration")
        sequence = match.group("sequence")
        hpstr = match.group("hyperparameters")

        # Parse the hyperparameters string into a dictionary
        hplookup = {}
        for param in hpstr.split(", "):
            key, value = param.split("=")
            hplookup[key] = float(value)

        hparams = HyperParams(
            learningrate=float(hplookup["learningrate"]),
            lrdecay=float(hplookup["lrdecay"]),
            tauRatings=float(hplookup["tauRatings"]),
            tauL2=float(hplookup["tauL2"]),
            depth=int(hplookup["depth"]),
            hiddenDims=int(hplookup["hiddenDims"]),
            queryDims=int(hplookup["queryDims"]),
            inducingPoints=int(hplookup["inducingPoints"])
        )

        vlosses = []
        for line in lines[1:]:
            match = vlpattern.search(line)
            if match:
                vlosses.append(float(match.group(1)))
        validationloss = min(vlosses)

        return hparams, validationloss

    def training(self, sequence: int, hparams: HyperParams):
        # Run the training in a manual subprocess of "train.py".
        # This is implemented manually instead of using a ProcessPool due to spurious hanging with the latter.

        scriptpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
        logpath = f"{self.logdir}/{self.title}/training_{self.iteration}_{sequence}.txt"
        trainlosspath = f"{self.logdir}/{self.title}/trainloss_{self.iteration}_{sequence}.txt"
        validationlosspath = f"{self.logdir}/{self.title}/validationloss_{self.iteration}_{sequence}.txt"
        modelpath = f"{self.netdir}/{self.title}/model_{self.iteration}_{sequence}_{{}}.pth"
        bestmodelpath = modelpath.replace("{}", "")  # best validationloss model by subprocess

        if self.resume and os.path.exists(bestmodelpath):
            # skip this training run for existing result
            hparams, validationloss = self.loadResults(logpath)
            print(f"Validation loss it{self.iteration} seq{sequence}: {validationloss}")
            return hparams, bestmodelpath, validationloss

        args = [
            "python3", scriptpath,
            self.listfile, self.featuredir, "-f", self.featurename,
            "--outfile", modelpath,
            "--trainlossfile", trainlosspath,
            "--validationlossfile", validationlosspath,
            "--batch-size", str(self.batchSize),
            "--steps", str(self.steps),
            "--epochs", str(self.epochs),
            "--learningrate", str(hparams.learningrate),
            "--lrdecay", str(hparams.lrdecay),
            "--patience", str(self.patience),
            "--tau-ratings", str(hparams.tauRatings),
            "--tau-l2", str(hparams.tauL2),
            "--depth", str(hparams.depth),
            "--hidden-dims", str(hparams.hiddenDims),
            "--query-dims", str(hparams.queryDims),
            "--inducing-points", str(hparams.inducingPoints)]

        if self.workers > 1:  # we cannot have every worker load the huge dataset
            args.append("--lowmemory")

        # before starting training, clean previous results
        os.path.exists(validationlosspath) and os.remove(validationlosspath)
        os.path.exists(bestmodelpath) and os.remove(bestmodelpath)

        # run training to the end
        with open(logpath, "w") as logfile:
            self.logMessage(logfile, f"HP Search it{self.iteration} seq{sequence} | {hparams}")
            process = subprocess.Popen(args, stdout=logfile)
        process.wait()

        # find min validationloss from file written by subprocess
        assert os.path.exists(validationlosspath), f"Loss file by training process not found: {validationlosspath}"
        assert os.path.exists(bestmodelpath), f"Model file by training process not found: {bestmodelpath}"

        with open(validationlosspath, "r") as file:
            validationloss = min(float(line.split(',')[0]) for line in file if line.strip())

        print(f"Validation loss it{self.iteration} seq{sequence}: {validationloss}")
        return hparams, bestmodelpath, validationloss

    def logMessage(self, logfile, message):
        print(message)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logfile.write(f"[{timestamp}] {message}\n")

if __name__ == "__main__":
    description = """
    Train strength models with different hyperparameters to find the best ones.
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
    optional_args.add_argument("-T", "--title", help="Subdirectory name for files produced by this search", type=str, default="search", required=False)
    optional_args.add_argument("-n", "--netdir", help="Directory name for trained models", type=str, default="nets", required=False)
    optional_args.add_argument("-l", "--logdir", help="Directory name for log files", type=str, default="logs", required=False)
    optional_args.add_argument("-b", "--batch-size", help="Minibatch size", type=int, default=100, required=False)
    optional_args.add_argument("-t", "--steps", help="Number of batches per epoch", type=int, default=100, required=False)
    optional_args.add_argument("-e", "--epochs", help="Nr of training epochs", type=int, default=5, required=False)
    optional_args.add_argument("-w", "--workers", help="Nr of concurrent worker processes", type=int, default=1, required=False)
    optional_args.add_argument("-r", "--resume", help="Reuse results of past training runs", action="store_true", required=False)
    optional_args.add_argument("-p", "--patience", help="Epochs without improvement before early stop", type=int, default=3, required=False)
    optional_args.add_argument("-s", "--samples", help="Nr of training runs per iteration", type=int, default=15, required=False)
    optional_args.add_argument("-i", "--iterations", help="Nr of hyperparameter search iterations", type=int, default=2, required=False)
    optional_args.add_argument("-d", "--decay", help="Decay of search scale", type=float, default=1, required=False)

    args = vars(parser.parse_args())
    main(args)
