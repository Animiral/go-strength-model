#!/usr/bin/env python3
# Usage: plots/search.py logs/search
"""
Visualize the marginal distributions p(L|t) of the validation loss L seen in the search under every hyperparameter t.
The data is taken from all training*.txt files in the argument directory.
"""

# import numpy as np
import argparse
import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import fontconfig
import re

def readHparams(logPath: str):
  # This function works both on the old log format (from preliminary hp search) and the current format.
  pattern1 = r"HP Search it(?P<iteration>\d+) seq(?P<sequence>\d+) \| HyperParams\((?P<hparams>[^\)]+)\)"
  pattern2 = r"HP Search it(?P<iteration>\d+) seq(?P<sequence>\d+) \| (?P<hparams>.+)"

  with open(logPath, "r") as file:
    line = file.readline()

  match = re.search(pattern1, line) or re.search(pattern2, line)
  if not match:
    raise ValueError(f"Failed to parse header of {logPath}")

  iteration = match.group("iteration")
  sequence = match.group("sequence")
  hparamstr = match.group("hparams")
  
  hparamdict = {}
  for param in re.split(",?\\s", hparamstr):
    key, value = param.split("=")
    hparamdict[key] = value

  # key replacement map
  remap = {k: k for k in ["learningrate", "lrdecay", "tauRatings", "tauL2", "depth", "hiddenDims", "queryDims", "inducingPoints"]}
  remap.update({
    "lr": "learningrate",
    "decay": "lrdecay",
    "l": "depth",
    "d": "hiddenDims",
    "dq": "queryDims",
    "m": "inducingPoints",
    "N": "N"  # window size; no longer in hp search
  })

  hparams = {}
  for name, vstr in hparamdict.items():
    name = remap[name]
    v = float(vstr) if name in ["learningrate", "lrdecay", "tauRatings", "tauL2"] else int(vstr)
    hparams[name] = v

  hparams["iteration"] = int(iteration)  # include this for plot marker
  return hparams

def readVloss(vlossPath: str):
  vlosses = []

  with open(vlossPath, "r") as file:
    for line in file.readlines():
        tokens = line.split(",")
        if tokens:
          vlosses.append(float(tokens[0]))

  return min(vlosses)

def findRuns(directory: str):
  runs = []

  for root, _, files in os.walk(directory):
    for file in files:
      if file.startswith("training") and file.endswith(".txt"):
        try:
          logPath = os.path.join(root, file)
          rundata = readHparams(logPath)
          vlossPath = logPath.replace("training", "validationloss")
          rundata["vloss"] = readVloss(vlossPath)
          runs.append(rundata)
        except ValueError as e:
          print(f"Error processing {file} ({e}), skip")
          continue
  
  return pd.DataFrame(runs)

def plot(df, zoom: bool = False):
  paramnames = [c for c in df.columns if c not in ["vloss", "iteration"]]
  titles = {
    "learningrate": "Learning Rate",
    "lrdecay": "Learning Rate Decay",
    "tauRatings": "Rating Loss Scaling Factor",
    "tauL2": "Regularization Factor",
    "depth": "Model Depth",
    "hiddenDims": "Hidden Dimensions",
    "queryDims": "Query Dimensions",
    "inducingPoints": "Inducing Points",
    "N": "Window Size"
  }

  fig, axs = plt.subplots(4, 2, figsize=fontconfig.big_figsize) # figsize=(20, 10))
  axs = axs.ravel()

  # colormap = plt.colormaps["tab10"]
  # colors = colormap.colors[:len(df)]
  colors = plt.cm.get_cmap("tab10", len(df)).colors
  shapes = {0: "o", 1: "s", 2: "D", 3: "v", 4: "X"}

  for i, param in enumerate(paramnames):
    ax = axs[i]
    for idx, row in df.iterrows():
      ax.scatter(row[param], row["vloss"], edgecolors=colors[idx], facecolors="none", marker=shapes[row["iteration"]])

    # ax.set_title(f"{titles[param]}")
    ax.set_xlabel(f"{titles[param]}")
    if True: # 0 == (i % 2):  # ylabels only on left hand side
      ax.set_ylabel("Min Validation Loss")
    if zoom:
      ax.set_ylim(0.578, 0.59)  # optional: hide outliers

    if param == "learningrate":
      ax.set_xscale("log")
      # ax.set_xlim(0, 0.01)
    if param == "depth":
      ax.set_xticks([1, 2, 3, 4, 5])

  # Hide the last subplot if there are fewer plots needed
  for i in range(len(paramnames), len(axs)):
    axs[i].axis("off")

  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  description = """
  Visualize the marginal distributions p(L|t) of the validation loss L seen in the search under every hyperparameter t.
  The data is taken from all training*.txt files in the argument directory.
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("-z", "--zoom", action="store_true", help="Restrict the y-axis to good candidates.")
  parser.add_argument("logdir", type=str, help="Path to the log directory of the hyperparameter search.")
  args = parser.parse_args()
  print(vars(args))

  df = findRuns(args.logdir)
  print("Preparing plots...")
  plot(df, args.zoom)
