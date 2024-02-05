"""
Given an input file listing Go matches, split them into a training set,
a validation set and a test set.
The split fractions can be provided as arguments.
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
        "-t", "--trainingFraction",
        default="0.8",
        type=lambda x: float(x) if 0.0 <= float(x) <= 1.0 else argparse.ArgumentTypeError(f"trainingFraction must be between 0 and 1"),
        help="Fraction of total records which should be given the test set marker",
    )
    parser.add_argument(
        "-v", "--validationFraction",
        default="0.1",
        type=lambda x: float(x) if 0.0 <= float(x) <= 1.0 else argparse.ArgumentTypeError(f"validationFraction must be between 0 and 1"),
        help="Fraction of total records which should be given the validation set marker",
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
        trainingFraction = args.trainingFraction
        validationFraction = args.validationFraction
        rowCount = len(rows)
        trainCount = round(rowCount * trainingFraction)
        validationCount = round(rowCount * validationFraction)
        testCount = rowCount - trainCount - validationCount

        print(f"Randomly splitting {len(rows)} rows {trainingFraction}/{validationFraction}/...: {trainCount} training, {validationCount} validation, {testCount} testing.")

        markers = list('T'*trainCount + 'V'*validationCount + 'E'*testCount)
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
