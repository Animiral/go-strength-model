"""
Given an input file listing Go matches, split them into a training set,
a validation set and a test set.
The split amounts can be provided as arguments.
Output the evaluation results to a new file or the same file with the assignment in the 'Set' column:
  - 'T' for training games
  - 'V' for validation games
  - 'E' for test games
"""

import argparse
import csv
import random

def eligibleRows(csvrows, modify: bool):
    """Mask out rows where one player is new in the system, giving us no prior info."""
    occurred = set()
    mask = []

    for row in csvrows:
        white, black = row["Player White"], row["Player Black"]
        if modify and "Set" in row and row["Set"] != "-":
            mask.append(False)
        elif white in occurred and black in occurred:
            mask.append(True)
        else:
            mask.append(False)
        occurred.add(white)
        occurred.add(black)

    return mask

def countRows(csvrows, setmarker: str, excludeMask):
    assert len(mask) == len(csvrows)
    return sum(1 for i, row in enumerate(csvrows) if row["Set"] == setmarker and not excludeMask[i])

def spread(markers, mask):
    """Distribute shuffled markers over eligible rows."""
    assert len(markers) == sum(mask)
    it = iter(markers)
    return ['-' if not el else next(it) for el in mask]

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
        default=False,
        type=bool,
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
        "-n", "--withNovice",
        default=False,
        type=bool,
        help="Mark even such records where (one of) the players occurs for the first time.",
    )
    args = parser.parse_args()
    print(vars(args))

    markerlookup = dict()
    if args.copy:
        with open(args.copy, 'r') as copyfile:
            copyreader = csv.DictReader(copyfile)
            copyrows = list(copyreader)
            markerlookup = { r["File"] : r["Set"] for r in copyrows }

    # Input CSV format (title row):
    # File,Player White,Player Black,Winner
    with open(args.input, 'r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        infile.close()

    if args.copy:
        assert not args.modify, "incompatible arguments: --copy and --modify"
        print(f"Copying training/validation/test set markers from {args.copy}.")

        for r in rows:
            r["Set"] = markerlookup.get(r["File"], "-")  # tolerate copy from small dataset to larger dataset
    else:
        trainingPart = args.trainingPart
        validationPart = args.validationPart

        # find all rows that we can mark
        mask = [True] * len(rows) if args.withNovice else eligibleRows(rows, args.modify)
        rowCount = sum(mask)

        # determine how many assignments to which set among eligible rows
        if args.trainingPart < 1:
            trainCount = int(round(rowCount * args.trainingPart))
        else:
            trainCount = int(args.trainingPart)
        if args.modify:
            preexisting = countRows(csvrows, "T", mask)
            assert preexisting <= trainCount, "Modification to reduce dataset size is not supported"
            trainCount -= preexisting

        if args.validationPart < 1:
            validationCount = int(round(rowCount * args.validationPart))
        else:
            validationCount = int(args.validationPart)
        if args.modify:
            preexisting = countRows(csvrows, "V", mask)
            assert preexisting <= validationCount, "Modification to reduce dataset size is not supported"
            validationCount -= preexisting

        if args.testPart is None:
            testCount = rowCount - trainCount - validationCount
        elif args.testPart < 1:
            testCount = int(round(rowCount * args.testPart))
        else:
            testCount = int(args.testPart)
        if args.modify:
            preexisting = countRows(csvrows, "E", mask)
            assert preexisting <= testCount, "Modification to reduce dataset size is not supported"
            testCount -= preexisting

        remainder = rowCount - trainCount - validationCount - testCount
        if remainder < 0:
            print(f"Not enough rows to mark them! T={trainCount}, V={validationCount}, E={testCount}, eligible rows={rowCount}/{len(rows)}")
            exit()

        print(f"Randomly splitting {rowCount} of {len(rows)} rows {trainingPart}/{validationPart}/...: {trainCount} training, {validationCount} validation, {testCount} testing.")

        markers = list('T'*trainCount + 'V'*validationCount + 'E'*testCount + '-'*remainder)
        random.shuffle(markers)
        markers = spread(markers, mask)

        for r, m in zip(rows, markers):
            r["Set"] = m

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
