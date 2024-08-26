"""
All-in-one script for extracting different bits of data from the dataset for plots etc.
"""

import argparse
import sys
import csv
import re
from moves_dataset import *

def loss_rating(listpath, featuredir, outpath):
  """
  Extract the CSV for the ratings vs points/winrate loss plots.
  """
  ds = MovesDataset(listpath, featuredir, marker="T", featurename="head", featurememory=False)
  with open(outpath, "w") as outfile:
      writer = csv.DictWriter(outfile, fieldnames=["rating", "wrloss", "ploss"])
      writer.writeheader()
      row = {"rating": 0.0, "wrloss": 0.0, "ploss": 0.0}
      for bx, wx, by, wy, _ in ds:
        row["rating"] = scale_rating(by)
        for x in bx:
          row["wrloss"] = x[4].item()
          row["ploss"] = x[5].item()
          writer.writerow(row)
        row["rating"] = scale_rating(wy)
        for x in wx:
          row["wrloss"] = x[4].item()
          row["ploss"] = x[5].item()
          writer.writerow(row)
        print(".", end="")

def rating(listpath, featuredir, outpath):
  """
  Extract just the ratings for the ratings density plot.
  """
  ds = MovesDataset(listpath, featuredir, marker="T", featurename="head", featurememory=False)
  with open(outpath, "w") as outfile:
      for _, _, by, wy, _ in ds:
        outfile.write(f"{scale_rating(by)}\n")
        outfile.write(f"{scale_rating(wy)}\n")

def gamelength(listpath, outpath):
  """
  Extract the lengths (in moves) of games that went to counting.
  """
  def get_sgf_moves(sgfpath):
    move_rex = re.compile(r";[WB]\[[^\]]", flags=re.IGNORECASE)  # matches moves like ";W[dq]", but not passes like ";B[]"
    with open(sgfpath, "r") as sgffile:
      length = len(re.findall(move_rex, sgffile.read()))
    return length

  counted_rex = re.compile(r"[WB]\+\d")  # matches game results like "W+30" or "B+2.5"
  with open(listpath, "r") as infile:
    with open(outpath, "w") as outfile:
      reader = csv.DictReader(infile)
      for row in reader:
        if "T" == row["Set"] and counted_rex.match(row["Winner"]):  # count only in training set
        # if counted_rex.match(row["Winner"]):
          length = get_sgf_moves(row["File"])
          outfile.write(f"{length}\n")

def basics(listpath):
  games = 0
  whitewins = 0
  with open(listpath, "r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
      games += 1
      if float(row["Score"]) < 0.5:
        whitewins += 1
  print(f"games: {games}, white wins: {whitewins}.")

if __name__ == "__main__":
  description = """
  All-in-one script for extracting different bits of data from the dataset for plots etc.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("command", choices=["loss_rating", "rating", "gamelength", "basics"], help="What data to get")
  parser.add_argument("listfile", type=str, help="CSV file listing games and labels")
  parser.add_argument("featuredir", type=str, help="directory containing extracted features")
  parser.add_argument("outpath", type=str, help="CSV output file path")
  args = vars(parser.parse_args())
  print(args)

  if "loss_rating" == args["command"]:
    loss_rating(args["listfile"], args["featuredir"], args["outpath"])

  if "rating" == args["command"]:
    rating(args["listfile"], args["featuredir"], args["outpath"])

  if "gamelength" == args["command"]:
    gamelength(args["listfile"], args["outpath"])

  if "basics" == args["command"]:
    basics(args["listfile"])

  print("Done.")
