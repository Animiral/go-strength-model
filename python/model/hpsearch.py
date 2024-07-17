#!/usr/bin/env python3
# Hyperparameter search algorithm.

import argparse
import os
import datetime
import copy
import math
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from moves_dataset import MovesDataset, MovesDataLoader, bradley_terry_score
from train import Training
from strengthnet import StrengthNet
from torch.optim.lr_scheduler import StepLR

device = "cuda"

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    featurename = args["featurename"]
    title = args["title"]
    netdir = args["netdir"]
    logdir = args["logdir"]
    batchSize = args["batch_size"]
    steps = args["steps"]
    epochs = args["epochs"]
    samples = args["samples"]
    broadIterations = args["broad_iterations"]
    fineIterations = args["fine_iterations"]

    print(f"Load training data from {listfile}")
    print(f"Load precomputed {featurename} features from {featuredir}")
    print(f"This search is titled \"{title}\"")
    print(f"Save networks in {netdir}/{title}")
    print(f"Save logs in {logdir}/{title}")
    print(f"Batch size: {batchSize}")
    print(f"Steps: {steps}")
    print(f"Epochs: {epochs}")
    print(f"Samples per iteration: {samples}")
    print(f"Number of broad search iterations: {broadIterations}")
    print(f"Number of fine search iterations: {fineIterations}")
    print(f"Device: {device}")

    trainData = MovesDataset(listfile, featuredir, "T", featurename=featurename)
    validationData = MovesDataset(listfile, featuredir, "V", featurename=featurename)
    search = HyperparamSearch(title, trainData, validationData, netdir, logdir, epochs, steps, batchSize, samples)
    logfile = open(f"{logdir}/{title}.txt", "w")

    def logMessage(message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}\n")
        logfile.write(f"[{timestamp}] {message}\n")

    model = None

    for i in range(broadIterations):
        logMessage(f"=== Hyperparameter Search: Broad Iteration {i} ===")
        hparams, model, validationloss = search.searchBroad()
        learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints = hparams
        logMessage(f"Best hparams (vloss={validationloss}) broad{i}: lr={learningrate} decay={lrdecay} " +
            f"N={windowSize} l={depth} d={hiddenDims} dq={queryDims} m={inducingPoints}")

    for i in range(fineIterations):
        logMessage(f"=== Hyperparameter Search: Fine Iteration {i} ===")
        hparams, model, validationloss = search.searchFine()
        learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints = hparams
        logMessage(f"Best hparams (vloss={validationloss}) fine{i}: lr={learningrate} decay={lrdecay} " +
            f"N={windowSize} l={depth} d={hiddenDims} dq={queryDims} m={inducingPoints}")

    if model:
        modelfile = f"{netdir}/{title}/model.pth"
        model.save(modelfile)

    logfile.close()

class HyperparamSearch:
    """Run multiple trainings with different hyperparameters."""

    def __init__(self, title: str, trainData: MovesDataset, validationData: MovesDataset,
                 netdir: str, logdir: str, epochs: int, steps: int, batchSize: int, trainingSamples: int):
        self.title = title
        os.makedirs(f"{netdir}/{title}", exist_ok=True)
        os.makedirs(f"{logdir}/{title}", exist_ok=True)
        self.iteration = 0
        self.trainData = trainData
        self.validationData = validationData
        self.netdir = netdir
        self.logdir = logdir
        self.epochs = epochs
        self.steps = steps
        self.batchSize = batchSize
        self.trainingSamples = trainingSamples
        hparams = (10**-3,   # learning rate
                   0.95,     # lr decay
                   400,      # window size
                   3,        # model depth
                   64,       # hidden dims
                   64,       # query dims
                   32)       # inducing points
        self.best = (hparams, None, float("inf"))

    def search(self, randomParams):
        samples = []
        for sequence in range(self.trainingSamples):
            hparams = randomParams()
            model, validationloss = self.training(sequence, hparams)
            samples.append((hparams, model, validationloss))
        self.best = min(samples + [self.best], key=lambda x: x[2])
        self.iteration += 1
        return self.best

    def searchBroad(self):
        return self.search(self.randomParamsBroad)

    def searchFine(self):
        return self.search(self.randomParamsFine)

    def randomParamsBroad(self):
        learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints = self.best[0]
        learningrate = learningrate * (10**random.uniform(-3, 3))
        lrdecay = lrdecay + random.uniform(-0.5, 0.5)
        windowSize = random.randint(10, 500)
        depth = random.randint(1, 5)
        hiddenDims = int(math.ceil(hiddenDims * (2**random.uniform(-2, 2))))
        queryDims = int(math.ceil(queryDims * (2**random.uniform(-2, 2))))
        inducingPoints = inducingPoints + random.randint(-32, 32)
        return self.clampParams(learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints)

    def randomParamsFine(self):
        learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints = self.best[0]
        learningrate = learningrate * (10**random.uniform(-0.3, 0.3))
        lrdecay = lrdecay + random.uniform(-0.05, 0.05)
        windowSize = windowSize + random.randint(-100, 100)
        depth = random.randint(1, 5)
        hiddenDims = int(math.ceil(hiddenDims * (2**random.uniform(-0.2, 0.2))))
        queryDims = int(math.ceil(queryDims * (2**random.uniform(-2, 2))))
        inducingPoints = inducingPoints + random.randint(-3, 3)
        return self.clampParams(learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints)

    def clampParams(self, learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints):
        learningrate = 10**-5 if learningrate < 10**-5 else 1 if learningrate > 1 else learningrate
        lrdecay = 0.9 if lrdecay < 0.9 else 1 if lrdecay > 1 else lrdecay
        windowSize = 10 if windowSize < 10 else 500 if windowSize > 500 else windowSize
        depth = 1 if depth < 1 else 5 if depth > 5 else depth
        hiddenDims = 8 if hiddenDims < 8 else 256 if hiddenDims > 256 else hiddenDims
        queryDims = 8 if queryDims < 8 else 256 if queryDims > 256 else queryDims
        inducingPoints = 1 if inducingPoints < 1 else 64 if inducingPoints > 64 else inducingPoints
        return learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints

    def training(self, sequence: int, hparams):
        logfile = open(f"{self.logdir}/{self.title}/training_{self.iteration}_{sequence}.txt", "w")
        trainlossfile = open(f"{self.logdir}/{self.title}/trainloss_{self.iteration}_{sequence}.txt", "w")
        validationlossfile = open(f"{self.logdir}/{self.title}/validationloss_{self.iteration}_{sequence}.txt", "w")

        learningrate, lrdecay, windowSize, depth, hiddenDims, queryDims, inducingPoints = hparams
        self.logMessage(logfile, f"HP Search it{self.iteration} seq{sequence} | lr={learningrate} decay={lrdecay} " +
            f"N={windowSize} l={depth} d={hiddenDims} dq={queryDims} m={inducingPoints}")

        validationLoader = MovesDataLoader(windowSize, self.validationData, batch_size=self.batchSize)
        model = StrengthNet(self.trainData.featureDims, depth, hiddenDims, queryDims, inducingPoints)
        model = model.to(device)

        def callback(model, e, trainloss, validationloss):
            self.epochResult(model, sequence, e, trainloss, validationloss, logfile, trainlossfile, validationlossfile)

        t = Training(callback, self.trainData, validationLoader,
                     self.epochs, self.steps, self.batchSize, learningrate, lrdecay, windowSize)
        bestmodel, validationloss = t.run(model)

        logfile.close()
        trainlossfile.close()
        validationlossfile.close()
        return bestmodel, validationloss

    def epochResult(self, model, sequence: int, e: int,
            trainloss, validationloss, logfile, trainlossfile, validationlossfile):
        self.logMessage(logfile, f"\tEpoch {e} error: training {trainloss[-1]:>8f}, validation {validationloss:>8f} \n")
        validationlossfile.write(f"{validationloss}\n")
        for loss in trainloss:
            trainlossfile.write(f"{loss}\n")
        modelfile = f"{self.netdir}/{self.title}/model_{self.iteration}_{sequence}_{e}.pth"
        model.save(modelfile)

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
    optional_args.add_argument("-d", "--title", help="Subdirectory name for files produced by this search", type=str, default="search", required=False)
    optional_args.add_argument("-n", "--netdir", help="Directory name for trained models", type=str, default="nets", required=False)
    optional_args.add_argument("-l", "--logdir", help="Directory name for log files", type=str, default="logs", required=False)
    optional_args.add_argument("-b", "--batch-size", help="Minibatch size", type=int, default=100, required=False)
    optional_args.add_argument("-t", "--steps", help="Number of batches per epoch", type=int, default=100, required=False)
    optional_args.add_argument("-e", "--epochs", help="Nr of training epochs", type=int, default=5, required=False)
    optional_args.add_argument("-s", "--samples", help="Nr of training runs per iteration", type=int, default=15, required=False)
    optional_args.add_argument("-i", "--broad-iterations", help="Nr of broad hyperparameter search iterations", type=int, default=2, required=False)
    optional_args.add_argument("-j", "--fine-iterations", help="Nr of fine hyperparameter search iterations", type=int, default=2, required=False)

    args = vars(parser.parse_args())
    main(args)
