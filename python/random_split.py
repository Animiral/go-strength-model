#!/usr/bin/env python3
"""
Usage: random_split.py [-h] [-i INPUT_PATH] [-c COPYFROM_PATH] [--modify] [-o OUTPUT_PATH] [-t TRAININGPART] [-v VALIDATIONPART]
                       [-e TESTPART] [-x EXHIBITIONPART] [--with-novice] [-a ADVANCE]

Given an input file listing Go matches, split them into a training set,
a validation set, a test set and, optionally, an exhibition set.
The split amounts can be provided as arguments.
Output the evaluation results to a new file or the same file with the assignment in the 'Set' column:
  - 'T' for training games
  - 'V' for validation games
  - 'E' for test games
  - 'X' for exhibition games
"""

import argparse
from enum import Enum
from typing import Optional
from dataclasses import dataclass
import csv
import random

# -----------------------------------------------

class Pool(Enum):
    UNASSIGN = 0    # assign to no set
    TRAINING = 1    # assign to training set
    VALIDATION = 2  # assign to validation set
    TEST = 3        # assign to test set
    EXHIBITION = 4  # assign to exhibition set
    ELIGIBLE = 5    # non-novice rows that we can use in validation and test
    TRAINABLE = 6   # low-noise rows that we can use in training set
    EXHIBITABLE = 7 # games between players with age 1 to 4 in the dataset

@dataclass
class PoolRow:
    row: dict
    pool: Pool

def markEligibles(prows: list[PoolRow]):
    """Find rows where both players are established in the system and put them in the eligible pool."""
    players = set()

    for prow in prows:
        white, black = prow.row["Player White"], prow.row["Player Black"]
        if white in players and black in players:
            prow.pool = Pool.ELIGIBLE
        players.add(white)
        players.add(black)

def markTrainables(prows: list[PoolRow], advance: Optional[int]):
    """
    Find rows which are suitable for training.
    These are rows in which the labels must be established by a number of games
    and they must agree with the score.
    """
    players = dict()

    for prow in reversed(prows):
        # update counts
        white, black = prow.row["Player White"], prow.row["Player Black"]
        wcount, bcount = players.get(white, 0), players.get(black, 0)
        wcount += 1
        bcount += 1
        players[white] = wcount
        players[black] = bcount

        # check for trainable
        if Pool.ELIGIBLE != prow.pool:
            continue  # non-eligible rows are not suitable for training

        if advance is None:
            prow.pool = Pool.TRAINABLE  # all eligible rows are suitable for training
            continue

        if wcount <= advance or bcount <= advance:
            continue  # not enough future history

        score = float(prow.row["Score"])
        wrating = float(prow.row["WhiteRating"])
        brating = float(prow.row["BlackRating"])
        if score < 0.5 and brating > wrating:
            continue  # white wins, but black labeled stronger
        if score > 0.5 and brating < wrating:
            continue  # black wins, but white labeled stronger

        prow.pool = Pool.TRAINABLE

def markExhibitables(prows: list[PoolRow], modify: bool):
    """
    Find rows which are suitable for exhibiting.
    These are rows in which both players have no more than 4 past games in the pool.
    """
    players = dict()

    for prow in prows:
        # update counts
        white, black = prow.row["Player White"], prow.row["Player Black"]
        wcount, bcount = players.get(white, 0), players.get(black, 0)
        wcount += 1
        bcount += 1
        players[white] = wcount
        players[black] = bcount

        # check for exhibitable
        if Pool.ELIGIBLE != prow.pool:
            continue  # non-eligible rows are not suitable for exhibition

        if wcount > 5 or bcount > 5:
            continue  # too much past history

        if modify and prow.row["Set"] not in ["-", "X"]:
            continue  # preserve other sets when modifying (this sometimes randomly costs us potential X games, but ok)

        prow.pool = Pool.EXHIBITABLE

def spreadRandom(prows: list[PoolRow], source: Pool, dest: Pool, count: int):
    """Randomly assign `count` rows from the `source` pool to the `dest` pool."""
    print(f"Randomly assign {count} rows from {source} to {dest}.")
    sourceCount = sum(source == prow.pool for prow in prows)
    if count > sourceCount:
        raise ValueError(f"Cannot mark {count} {source} rows as {dest}: only {sourceCount} available.")

    markers = [dest]*count + [source]*(sourceCount-count)
    random.shuffle(markers)
    markers = iter(markers)

    for prow in prows:
        if source == prow.pool:
            prow.pool = next(markers)

def split(rows: list[dict], modify: bool, trainingPart: float, validationPart: float, testPart: Optional[float], exhibitionPart: float, advance: Optional[int], withNovice: bool):
    if withNovice:  # all rows are eligible
        prows = [PoolRow(row=r, pool=Pool.ELIGIBLE) for r in rows]
    else:
        prows = [PoolRow(row=r, pool=Pool.UNASSIGN) for r in rows]
        markEligibles(prows)

    # determine number of assignments and to which set among eligible rows
    eligibleCount = sum(Pool.ELIGIBLE == prow.pool for prow in prows)
    if args.trainingPart < 1:
        trainGoal = int(round(eligibleCount * args.trainingPart))
    else:
        trainGoal = int(args.trainingPart)
    if args.validationPart < 1:
        validationGoal = int(round(eligibleCount * args.validationPart))
    else:
        validationGoal = int(args.validationPart)
    if args.testPart is None:
        testGoal = eligibleCount - trainGoal - validationGoal
    elif args.testPart < 1:
        testGoal = int(round(eligibleCount * args.testPart))
    else:
        testGoal = int(args.testPart)
    if args.exhibitionPart < 1:
        exhibitionGoal = int(round(eligibleCount * args.exhibitionPart))
    else:
        exhibitionGoal = int(args.exhibitionPart)

    print(f"Randomly splitting {eligibleCount} eligible of {len(rows)} rows: {trainGoal} training, " +
          f"{validationGoal} validation, {testGoal} testing, {exhibitionGoal} exhibition.")

    # distribute training games among trainable rows
    markTrainables(prows, advance)
    print(f"Number of rows suitable for training: {sum(Pool.TRAINABLE == prow.pool for prow in prows)}.")
    # remember training set assignments and go from there
    for prow in prows:
        if modify and Pool.TRAINABLE == prow.pool and "T" == prow.row["Set"]:
            prow.pool = Pool.TRAINING
    trainCount = sum(Pool.TRAINING == prow.pool for prow in prows)

    if trainCount > trainGoal:
        spreadRandom(prows, source=Pool.TRAINING, dest=Pool.TRAINABLE, count=trainCount - trainGoal)
    if trainCount < trainGoal:
        spreadRandom(prows, source=Pool.TRAINABLE, dest=Pool.TRAINING, count=trainGoal - trainCount)

    # distribute exhibition games among exhibitable rows
    if exhibitionGoal > 0:
        for prow in prows:
            if Pool.TRAINABLE == prow.pool:  # we don't care about trainable anymore; only about eligible
                prow.pool = Pool.ELIGIBLE
        markExhibitables(prows, modify)
        print(f"Number of rows suitable for exhibition: {sum(Pool.EXHIBITABLE == prow.pool for prow in prows)}.")
        # remember exhibition set assignments and go from there
        for prow in prows:
            if modify and Pool.EXHIBITABLE == prow.pool and "X" == prow.row["Set"]:
                prow.pool = Pool.EXHIBITION
        exhibitionCount = sum(Pool.EXHIBITION == prow.pool for prow in prows)
        if exhibitionCount > exhibitionGoal:
            spreadRandom(prows, source=Pool.EXHIBITION, dest=Pool.EXHIBITABLE, count=exhibitionCount - exhibitionGoal)
        if exhibitionCount < exhibitionGoal:
            spreadRandom(prows, source=Pool.EXHIBITABLE, dest=Pool.EXHIBITION, count=exhibitionGoal - exhibitionCount)

    # distribute validation&test games among eligible rows
    for prow in prows:
        if Pool.TRAINABLE == prow.pool or Pool.EXHIBITABLE == prow.pool:  # we don't care about those anymore; only about eligible
            prow.pool = Pool.ELIGIBLE
        if modify and Pool.ELIGIBLE == prow.pool and "V" == prow.row["Set"]:
            prow.pool = Pool.VALIDATION
        if modify and Pool.ELIGIBLE == prow.pool and "E" == prow.row["Set"]:
            prow.pool = Pool.TEST
    validationCount = sum(Pool.VALIDATION == prow.pool for prow in prows)
    testCount = sum(Pool.TEST == prow.pool for prow in prows)

    if validationCount > validationGoal:
        spreadRandom(prows, source=Pool.VALIDATION, dest=Pool.ELIGIBLE, count=validationCount - validationGoal)
    if validationCount < validationGoal:
        spreadRandom(prows, source=Pool.ELIGIBLE, dest=Pool.VALIDATION, count=validationGoal - validationCount)
    if testCount > testGoal:
        spreadRandom(prows, source=Pool.TEST, dest=Pool.ELIGIBLE, count=testCount - testGoal)
    if testCount < testGoal:
        spreadRandom(prows, source=Pool.ELIGIBLE, dest=Pool.TEST, count=testGoal - testCount)

    # assign set by pool (all remaining eligible rows become "-"s)
    for prow in prows:
        prow.row["Set"] = {
            Pool.UNASSIGN: "-",
            Pool.ELIGIBLE: "-",
            Pool.EXHIBITABLE: "-",
            Pool.TRAINING: "T",
            Pool.VALIDATION: "V",
            Pool.TEST: "E",
            Pool.EXHIBITION: "X"
        }[prow.pool]

if __name__ == "__main__":
    description = """
    Add or modify the “Set” column to random distribution of T/V/E according to split.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-i", "--input",
        default="games.csv",
        type=str,
        help="Path to the CSV file listing the SGF files and players.",
    )
    parser.add_argument(
        "-c", "--copy",
        default="",
        type=str,
        help="Path to the CSV file which already holds the set information to copy to the output.",
    )
    parser.add_argument(
        "-m", "--modify",
        action="store_true",
        default=False,
        help="Adapt existing set assignments to the given numbers, changing as few rows as possible.",
    )
    parser.add_argument(
        "-o", "--output",
        default="",
        type=str,
        help="Path to the resulting CSV file with set markers.",
    )
    parser.add_argument(
        "-t", "--trainingPart",
        default="0.8",
        type=float,
        help="Part of total records which should be given the training set marker",
    )
    parser.add_argument(
        "-v", "--validationPart",
        default="0.1",
        type=float,
        help="Part of total records which should be given the validation set marker",
    )
    parser.add_argument(
        "-e", "--testPart",
        default=None,
        type=float,
        help="Part of total records which should be given the test set marker",
        required=False
    )
    parser.add_argument(
        "-x", "--exhibitionPart",
        default=0.0,
        type=float,
        help="Part of total records which should be given the exhibition set marker",
        required=False
    )
    parser.add_argument(
        "-n", "--with-novice",
        action="store_true",
        default=False,
        help="Mark even such records where (one of) the players occurs for the first time.",
    )
    parser.add_argument(
        "-a", "--advance",
        default=None,
        type=int,
        help="Require this number of future games for both players in training games.",
    )
    args = parser.parse_args()
    print(vars(args))

    # Input CSV format (title row):
    # File,Player White,Player Black,Winner
    with open(args.input, 'r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if args.copy:
        markerlookup = dict()

        with open(args.copy, 'r') as copyfile:
            copyreader = csv.DictReader(copyfile)
            copyrows = list(copyreader)
            markerlookup = { r["File"] : r["Set"] for r in copyrows }

        assert not args.modify, "incompatible arguments: --copy and --modify"
        print(f"Copying set markers from {args.copy}.")

        for r in rows:
            r["Set"] = markerlookup.get(r["File"], "-")  # tolerate copy from small dataset to larger dataset
        del markerlookup
    else:
        print(f"Assigning set markers for {len(rows)} rows from {args.input}.")
        split(rows, args.modify, args.trainingPart, args.validationPart, args.testPart, args.exhibitionPart, args.advance, args.with_novice)

    # write output CSV file
    outpath = args.output
    if "" == outpath:
        outpath = args.input
    with open(outpath, 'w') as outfile:
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        outfile.close()

    print("Finished writing output file.")
