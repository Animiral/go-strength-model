#!/usr/bin/env python3
# Evaluate strength model on marked games in dataset.

import argparse
import torch
from moves_dataset import MovesDataset, bradley_terry_score, scale_rating
from strengthnet import StrengthNet

device = "cuda"

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    modelfile = args["modelfile"]
    featurename = args["featurename"]
    outfile = args["outfile"]  # output as CSV like listfile, but only evaluated games and with Predicted* columns
    setmarker = args["setmarker"]
    if args["raw"]:
        scale = lambda x: x
    else:
        scale = scale_rating

    print(f"Evaluate games from {listfile} marked '{setmarker}'")
    print(f"Load precomputed {featurename} features from {featuredir}")
    print(f"Save results to {outfile}")
    print(f"Device: {device}")

    data = MovesDataset(listfile, featuredir, setmarker, featurename=featurename)
    model = StrengthNet.load(modelfile).to(device)

    for i, game in enumerate(data.marked):
        game.black.predictedRating, game.white.predictedRating, game.predictedScore = evaluate(data, i, model, scale)

    data.write(outfile)

def evaluate(data: MovesDataset, i: int, model: StrengthNet, scale):
    model.eval()
    with torch.no_grad():
        bx, wx, _, _, _ = data[i]
        bx, wx = bx.to(device), wx.to(device)
        bpred, wpred = model(bx).item(), model(wx).item()
        spred = bradley_terry_score(bpred, wpred)
        bpred, wpred = scale(bpred), scale(wpred)
    return bpred, wpred, spred

if __name__ == "__main__":
    description = """
    Evaluate strength model on marked games in dataset.
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
    required_args.add_argument('modelfile', help='Neural network weights file')
    optional_args.add_argument('-f', '--featurename', help='Type of features to train on', type=str, default='pick', required=False)
    optional_args.add_argument('-o', '--outfile', help='CSV file for output games and predictions', type=str, required=False)
    optional_args.add_argument('-m', '--setmarker', help='Marker for subset to evaluate', type=str, default='E', required=False)
    optional_args.add_argument('-r', '--raw', help='Do not apply scaling to model output', action="store_true", required=False)

    args = vars(parser.parse_args())
    main(args)
