import argparse
import math
import numpy as np
from scipy.stats import norm
from model.moves_dataset import MovesDataset
import pdb

def predict(blackPloss, whitePloss):
    if 0 == len(blackPloss) or 0 == len(whitePloss):
        return .5  # no data for prediction

    gamelength = 100  # assume 100 moves per player for an average game
    bMean = np.mean(blackPloss)
    bVar = np.var(blackPloss, ddof=1)
    wMean = np.mean(whitePloss)
    wVar = np.var(whitePloss, ddof=1)
    epsilon = 0.000001  # avoid div by 0
    z = math.sqrt(gamelength) * (wMean - bMean) / math.sqrt(bVar + wVar + epsilon)
    return norm.cdf(z)

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    outfile = args["outfile"]  # output as CSV like listfile, but only evaluated games and with Predicted* columns
    setmarker = args["setmarker"]

    print(f"Evaluate games from {listfile} marked '{setmarker}'")
    print(f"Load precomputed features from {featuredir}")
    print(f"Save results to {outfile}")

    data = MovesDataset(listfile, featuredir, setmarker, featurename="head")
    for i, game in enumerate(data.marked):
        bx, wx, _, _, score = data[i]
        # in head features, points loss is channel 5
        bx = bx[:,5].tolist()
        wx = wx[:,5].tolist()
        print(f"Predict {game.sgfPath}, {len(bx)} black and {len(wx)} white samples.")
        p = predict(bx, wx)
        # The normal distribution function tends to max out quickly due to float limits.
        # Clamp the probability to avoid overconfidence and infinitely wrong guesses.
        p = max(0.0000001, min(0.9999999, p))
        game.predictedScore = p

    data.write(outfile)

if __name__ == "__main__":
    description = """
    Evaluate the stochastic model on a dataset.
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
    required_args.add_argument('outfile', help='CSV output file path')
    optional_args.add_argument('-m', '--setmarker', help='Marker for subset to evaluate', type=str, default='E', required=False)

    args = vars(parser.parse_args())
    main(args)
