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
        print(f"Copying training/validation/test set markers from {args.copy}.")

        for r in rows:
            r["Set"] = markerlookup[r["File"]]
    else:
        trainingPart = args.trainingPart
        validationPart = args.validationPart
        rowCount = len(rows)
        trainCount = int(round(rowCount * args.trainingPart) if args.trainingPart < 1 else args.trainingPart)
        validationCount = int(round(rowCount * args.validationPart) if args.validationPart < 1 else args.validationPart)
        if args.testPart is None:
            testCount = rowCount - trainCount - validationCount
        else:
            testCount = int(round(rowCount * args.testPart) if args.testPart < 1 else args.testPart)
        remainder = rowCount - trainCount - validationCount - testCount
        if remainder < 0:
            print(f"Not enough rows to mark them! T={trainCount}, V={validationCount}, E={testCount}, rows={rowCount}")
            exit()

        print(f"Randomly splitting {len(rows)} rows {trainingPart}/{validationPart}/...: {trainCount} training, {validationCount} validation, {testCount} testing.")

        markers = list('T'*trainCount + 'V'*validationCount + 'E'*testCount + '-'*remainder)
        random.shuffle(markers)

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
