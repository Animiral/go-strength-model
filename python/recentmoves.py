# Determine the recent move set for all the marked games in a list file and write them to the feature dir
import argparse
import os
import csv
from model.moves_dataset import MovesDataset

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    marker = args["marker"]

    print(f"Load games data from {listfile}, type {marker}")
    print(f"Store recent moves specs in {featuredir}")

    dataset = MovesDataset(listfile, featuredir, marker)
    print(f"Loaded {len(dataset.games)} games, {len(dataset)} of type {marker}.")
    dataset.writeRecentMoves()

if __name__ == "__main__":
    description = """
    Extract recent moves for every marked game from dataset.
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
    required_args.add_argument('featuredir', help='directory containing extracted recent move lists')
    optional_args.add_argument('-m', '--marker', help='set marker of games to select', type=str, default='T', required=False)

    args = vars(parser.parse_args())
    main(args)
