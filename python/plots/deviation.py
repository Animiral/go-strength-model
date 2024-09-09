#!/usr/bin/env python3
# Usage: plots/deviation.py csv/games_glicko.csv
# 1. determine age of players in the dataset (how many games they have played to that point)
# 2. plot their glicko deviation value
# 3. plot ratings histogram over time
"""
Given an input CSV file with Glicko-2 analysis,
show the decline in RD (rating deviation) with every game.
"""
import sys
import csv
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from rank import *
import fontconfig

max_age_dev = 50  # for deviation
max_age_ranks = 15  # for rank distribution
num_samples = 20  # per age group

def read_csv(path):
  global max_age_dev
  age = {}  # for each player, how many times we have seen them
  rds = {}  # dict of list of tuples, age -> [(rating, RD)]

  with open(path, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
      white_name = row["Player White"]
      white_rating = float(row["WhiteRating"])
      white_deviation = float(row["WhiteDeviation"])
      white_age = age.get(white_name, 0)
      white_age += 1
      age[white_name] = white_age
      if white_age <= max_age_dev:
        entry = rds.setdefault(white_age, [])
        entry.append((white_rating, white_deviation))

      black_name = row["Player Black"]
      black_rating = float(row["BlackRating"])
      black_deviation = float(row["BlackDeviation"])
      black_age = age.get(black_name, 0)
      black_age += 1
      age[black_name] = black_age
      if black_age <= max_age_dev:
        entry = rds.setdefault(black_age, [])
        entry.append((black_rating, black_deviation))

  return rds

def shrink(rds):
  """Reduce the amount of data in the dict by determining a representative sample and the mean deviation"""
  global num_samples
  meandev = {}
  hist = {}  # histogram by rating class
  for k in rds:
    ranks = np.array([to_rank(rating) for rating, _ in rds[k]])
    ranks = np.clip(ranks, 10, 38)  # 20-kyu - 9-dan
    ranks = np.round(ranks)
    ranks, counts = np.unique(ranks, return_counts=True)
    hist[k] = dict(zip(ranks, counts))
    meandev[k] = np.mean([deviation for _, deviation in rds[k]])
    rds[k] = random.sample(rds[k], num_samples)
  return rds, meandev, hist

def plot_deviation(rds, meandev):
  global max_age_dev
  low_color = '#FFB6C1'  # pink
  medium_color = '#77DD77'  # green
  high_color = '#AEC6CF'  # blue
  overall_color = '#707070'  # grey

  x, y = zip(*((age, deviation) for age, data in rds.items() for (_, deviation) in data))
  plt.figure(figsize=fontconfig.ideal_figsize)

  # points
  plt.scatter(x, y, alpha=0.1)
  plt.gca().set_xlim(0, max_age_dev+1)
  # plt.plot(x, y, color=overall_color, linewidth=2, linestyle='--')

  # line
  plt.plot(meandev.keys(), meandev.values(), linestyle='-')

  plt.xlabel('Games')
  plt.ylabel('Rating Deviation')
  plt.title('Increase of Glicko-2 Confidence Over Time')
  # plt.legend()
  plt.show()

def plot_ratings(hist):
  global max_age_ranks

  # max_count = max(max(rankhist.values()) for rankhist in hist.values())
  plt.figure(figsize=fontconfig.ideal_figsize)

  for age in hist:
    if age > max_age_ranks:
      continue

    rankhist = hist[age]
    max_count = max(rankhist.values())

    for rank, count in rankhist.items():
      c = cm.Blues(0.2 + 0.8 * count / max_count)
      plt.bar(age, 1, bottom=rank, color=c, width=0.8*count / max_count)  #, alpha=0.3 + 0.7*count / max_count

  axes = plt.gca()
  axes.set_xlim(0, max_age_ranks+1)
  rank_ticks = np.array([10, 15, 20, 25, 30, 34, 38])
  axes.set_yticks(rank_ticks)
  axes.set_ylim(9, 39)
  axes.set_yticklabels(rankintstr(r) for r in rank_ticks)

  # rating axis
  rating_axis = axes.twinx()
  rating_ticks = [to_rating(r) for r in rank_ticks]
  rating_axis.set_yticks(rank_ticks)
  rating_axis.set_ylim(axes.get_ylim()[0], axes.get_ylim()[1])
  rating_axis.set_yticklabels(f'{r:.0f}' for r in rating_ticks)
  rating_axis.set_ylabel('Rating')

  axes.set_xlabel('Games')
  axes.set_ylabel('Rank')
  plt.title('Rank Distribution Over Time')
  # plt.legend()
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f'Read data from {path}...')
  rds = read_csv(path)
  rds, meandev, hist = shrink(rds)
  plot_deviation(rds, meandev)
  plot_ratings(hist)
