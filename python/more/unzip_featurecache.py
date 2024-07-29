#!/usr/bin/env python3
"""
Usage: unzip_featurecache.py [-h] [-f FEATURENAME] [-i] [-r] listfile featuredir

Unzip every training/validation/test feature file in the featurecache, which speeds
up future training runs especially if they run without memory caching in the dataset
or in different processes (hyperparameter search).
"""
import argparse
import csv
import os
import numpy as np
from model.moves_dataset import load_features_from_zip

def main(listpath: str, featuredir: str, featureName: str, ignoreErrors: bool, recompute: bool):
  print(f"Load dataset from {listpath}")
  print(f"Features in {featuredir}")
  print(f"Unzip {featureName} features")

  with open(listpath, "r") as listfile:
    reader = csv.DictReader(listfile)
    sgfPaths = [r["File"] for r in reader if r["Set"] in ["T", "V", "E"]]

  for sgfPath in sgfPaths:
    for player in ["Black", "White"]:
      basePath, _ = os.path.splitext(sgfPath)
      zipPath = f"{featuredir}/{basePath}_{player}Recent.zip"
      npyPath = f"{featuredir}/{basePath}_{player}Recent_{featureName}.npy"

      if not recompute and os.path.exists(npyPath):
        continue  # do not duplicate extract

      try:
        features = load_features_from_zip(zipPath, featureName)
        features = features.numpy()
        np.save(npyPath, features)
        print(f"{zipPath} -> {npyPath} ({features.shape})")
      except Exception as e:
        print(f"Error unzipping {zipPath}: {e}")
        if not ignoreErrors:
          return

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Unzip every training/validation/test feature file in the featurecache.")
  parser.add_argument("listfile", help="CSV file listing games and labels")
  parser.add_argument("featuredir", help="Directory containing extracted features")
  parser.add_argument("-f", "--featurename", help="Type of features to unzip", type=str, default="pick", required=False)
  parser.add_argument("-i", "--ignore-errors", help="If an error occurs in one file, continue anyway", action="store_true", required=False)
  parser.add_argument("-r", "--recompute", help="Unzip files even when the unzipped file already exists", action="store_true", required=False)
  args = parser.parse_args()

  main(args.listfile, args.featuredir, args.featurename, args.ignore_errors, args.recompute)
