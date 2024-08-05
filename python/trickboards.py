#!/usr/bin/env python
# Create strings from the initial positions of trickplay SGFs to feed into KataGo's genboard utility.

import os
import argparse
from sgfmill import sgf, boards, sgf_moves

def board_to_str(board: boards.Board, cutoff: tuple[int, int]) -> str:
    cutoff_row, cutoff_col = cutoff
    board_str = ""
    for row in range(19):
        line = []
        for col in range(19):
            stone = board.get(row, col)
            if row > cutoff_row or col < cutoff_col:
                line.append("?")
            elif stone is None:
                line.append(".")
            elif stone == "b":
                line.append("x")
            elif stone == "w":
                line.append("o")
        board_str += " ".join(line)
    return board_str

def str_to_board(boardstr: str) -> boards.Board:
    lines = boardstr.strip().split("\n")
    blacks = []
    whites = []

    for row, line in enumerate(lines):
        chars = line.strip().split()  # stones are space-separated by genboard
        for col, char in enumerate(chars):
            if "X" == char:
                blacks.append((row, col))
            elif "O" == char:
                whites.append((row, col))

    # node = sgf.Sgf_node()
    # node.set_setup_stones(blacks, whites, [])
    # return node
    board = boards.Board(19)
    board.apply_setup(blacks, whites, [])
    return board

def combine_game(initial_board: boards.Board, moves: sgf.Tree_node) -> sgf.Sgf_game:
    game = sgf.Sgf_game(19)
    sgf_moves.set_initial_position(game, initial_board)
    target = game.get_root()
    while moves is not None:
        target = target.new_child()
        target.set_move(*moves.get_move())
        moves = moves[0] if len(moves) else None
    return game

def extract(sgf_path: str, output_path: str, cutoff: tuple[int, int]):
    cutoff_row, cutoff_col = cutoff
    print(f"Processing {sgf_path}, cutoff={(cutoff_col, cutoff_row)} to {output_path}")

    with open(sgf_path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    # get initial position
    root = game.get_root()
    board = boards.Board(19)
    board.apply_setup(*root.get_setup_stones())

    board_str = board_to_str(board, cutoff)

    with open(output_path, "w") as f:
        f.write(board_str)

def merge(sgf_path: str, completed_path: str, success_path: str, failure_path: str):
    print(f"Merge {sgf_path} and {completed_path} -> {success_path}, {failure_path}")

    with open(sgf_path, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    root = game.get_root()
    with open(completed_path, "r") as f:
        boardstr = f.read()
    board = str_to_board(boardstr)
    sgf_moves.set_initial_position(game, board)

    success_game = combine_game(board, root[0])
    with open(success_path, "wb") as f:
        f.write(success_game.serialise())

    failure_game = combine_game(board, root[1])
    with open(failure_path, "wb") as f:
        f.write(failure_game.serialise())

def main(args):
    assert args.command in ["extract", "merge"]

    cutoff_row = ord(args.cutoff[1]) - ord('a')
    cutoff_col = ord(args.cutoff[0]) - ord('a')
    assert 0 <= cutoff_row <= 18
    assert 0 <= cutoff_col <= 18
    cutoff = (cutoff_row, cutoff_col)

    for path in os.listdir(args.sgfdir):
        if path.startswith("problem") and path.endswith(".sgf"):
            digitIdx = next((i for i, char in enumerate(path) if char.isdigit()), 0)
            fileSeq = path[digitIdx:].split('.')[0]
            boardpath = f"board{fileSeq}.txt"
            completedpath = f"completed{fileSeq}.txt"
            successpath = f"success{fileSeq}.sgf"
            failurepath = f"failure{fileSeq}.sgf"

            path, boardpath, completedpath, successpath, failurepath = map(
                lambda p: os.path.join(args.sgfdir, p), (path, boardpath, completedpath, successpath, failurepath))

            if "extract" == args.command:
                extract(path, boardpath, cutoff)
            elif "merge" == args.command:
                merge(path, completedpath, successpath, failurepath)

if __name__ == "__main__":
    description = """
    Create strings from the initial positions of trickplay SGFs to feed into KataGo's genboard utility.
    """

    parser = argparse.ArgumentParser(description=description,add_help=False)
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="show this help message and exit")
    parser.add_argument("command", type=str, help="extract|merge")
    parser.add_argument("sgfdir", type=str, help="Directory with SGF files to extract")
    parser.add_argument("-c", "--cutoff", type=str, default="hj", required=False, help="Smallest SGF coordinates of the known-region rectangle")

    args = parser.parse_args()
    main(args)
