"""
Given an input file listing the length of games (in moves),
plot them in a chart that shows their distribution.
"""

import sys
import numpy as np
import statistics
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import fontconfig

def read_file(path):
  with open(path, "r") as file:
    lengths = [int(line) for line in file]
  return lengths

def density(els):
  # kde = gaussian_kde(els)
  # x = np.linspace(100, 400, 300)
  # y = kde(x)
  x = list(range(100, 401))
  y = [els.count(i) for i in x]
  return x, y

def plot(x, y, mean_x):
  plt.figure(figsize=fontconfig.ideal_figsize)
  plt.plot(x, y)
  # mean indicator
  plt.axvline(mean_x, linestyle="--", linewidth=1, zorder=3)
  plt.xlabel("Game Moves")
  plt.ylabel("Count")
  plt.title("Distribution of Game Moves")
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f"Read data from {path}...")
  x = read_file(path)
  mean_x = statistics.mean(x)
  print(f"data count: {len(x)}")
  print(f"mean: {mean_x}")
  print(f"median: {statistics.median(x)}")
  print(f"mode: {statistics.mode(x)}")
  print(f"min-max: {min(x)}-{max(x)}")

  x, y = density(x)
  print("Preparing plot...")
  plot(x, y, mean_x)
