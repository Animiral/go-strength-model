# Print contents of a recent move .npz
import argparse
import os
import numpy as np
import torch


def print_tensor_board(black_stones, white_stones, last_move, next_move):
    def get_piece(x,y):
        if next_move[x,y] > 0:
            return 'A '
        elif black_stones[x,y] > 0:
            if last_move[x,y] > 0:
                return 'X '
            else:
                return 'x '
        elif white_stones[x,y] > 0:
            if last_move[x,y] > 0:
                return 'O '
            else:
                return 'o '
        elif (x == 3 or x == 19/2 or x == 19-1-3) and (y == 3 or y == 19/2 or y == 19-1-3):
            return '* '
        else:
            return '. '

    print("\n".join("".join(get_piece(x,y) for x in range(19)) for y in range(19)), "\n")

def main(args):
    recentfile = args["recentfile"]
    print(f"Load recent move data from {recentfile}")
    with np.load(recentfile) as npz:
        binaryInputNCHW = npz["binaryInputNCHW"]
        locInputNCHW = npz["locInputNCHW"]
        globalInputNC = npz["globalInputNC"]
    del npz

    binaryInputNCHW = torch.from_numpy(binaryInputNCHW) #.to(device)
    locInputNCHW = torch.from_numpy(locInputNCHW) #.to(device)
    globalInputNC = torch.from_numpy(globalInputNC) #.to(device)

    for b, l, g in zip(binaryInputNCHW, locInputNCHW, globalInputNC):
        print_tensor_board(b[1,:,:], b[2,:,:], b[9,:,:], l[0])
        # print(f"binaryInputNCHW shape: {b.shape}")
        # print(f"locInputNCHW shape: {l.shape}")
        # print(f"globalInputNC shape: {g.shape}")

if __name__ == "__main__":
    description = """
    Print contents of a recent move .npz for debugging.
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
    required_args.add_argument('recentfile', help='npz file containing an input tensor of recent moves')

    args = vars(parser.parse_args())
    main(args)
