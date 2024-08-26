"""
Given an input CSV file from a rating system evaluation,
determine the last rating predictions for every player ("final rating") and plot their distribution.
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from rank import *
import fontconfig

def read_list(path):
  with open(path, "r") as file:
    x = [float(line) for line in file]
  return x

def density(x):
  # estimate distribution
  df = gaussian_kde(x)
  x = np.linspace(min(x), max(x), 100)
  y = df(x)
  return x, y

def plot(x, y):
  low_color = '#FFB6C1'  # pink
  medium_color = '#77DD77'  # green
  high_color = '#AEC6CF'  # blue
  overall_color = '#707070'  # grey

  plt.figure(figsize=fontconfig.ideal_figsize)
  plt.plot(x, y, color=overall_color, linewidth=2, linestyle='--')
  plt.xlabel('Rating')
  plt.ylabel('Density')

  ddk = to_rank(x) <= 20
  sdk = (to_rank(x) > 20) & (to_rank(x) < 30)
  dan = to_rank(x) >= 30
  plt.fill_between(x[ddk], 0, y[ddk], color=low_color, alpha=0.3, label='10–kyu-25–kyu')
  plt.fill_between(x[sdk], 0, y[sdk], color=medium_color, alpha=0.3, label='9–kyu-1–kyu')
  plt.fill_between(x[dan], 0, y[dan], color=high_color, alpha=0.3, label='1–dan-9–dan')

  if x[ddk].size > 0:
    plt.text((x[ddk][-1]+x[ddk][0]) / 2, 0.00005, "DDK", horizontalalignment='center', size=20, color=low_color)
  if x[sdk].size > 0:
    plt.text((x[sdk][-1]+x[sdk][0]) / 2, 0.00005, "SDK", horizontalalignment='center', size=20, color=medium_color)
  if x[dan].size > 0:
    plt.text((x[dan][-1]+x[dan][0]) / 2, 0.00005, "DAN", horizontalalignment='center', size=20, color=high_color)

  # 2nd x-axis tick set for ranks
  axes = plt.gca()
  rank_axis = axes.twiny()
  rank_ticks = np.array([0, 10, 20, 25, 30, 34, 39])
  rank_axis.set_xticks(to_rating(rank_ticks))
  rank_axis.set_xlim(axes.get_xlim()[0], axes.get_xlim()[1])
  rank_axis.set_xticklabels(rankintstr(r) for r in rank_ticks)
  rank_axis.set_xlabel('Rank')

  plt.title('Distribution of Final Glicko-2 Rating')
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f'Read data from {path}...')
  # x = read_csv(path)
  x = read_list(path)
  x, y = density(x)
  plot(x, y)
