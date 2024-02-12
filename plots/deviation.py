"""
Given an input CSV file with Glicko-2 analysis,
show the decline in RD (rating deviation) with every game.
"""
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def read_csv(path):
  counter = {}  # for each player, how many times we have seen them
  rds = []  # list of tuples (occurrence count, RD)

  with open(path, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
      white_name = row["Player White"]
      white_counter = counter.get(white_name, 0)
      white_counter += 1
      rds.append((white_counter, float(row["WhiteDeviation"])))
      counter[white_name] = white_counter

      black_name = row["Player Black"]
      black_counter = counter.get(black_name, 0)
      black_counter += 1
      rds.append((black_counter, float(row["BlackDeviation"])))
      counter[black_name] = black_counter

  return rds

def plot(rds):
  low_color = '#FFB6C1'  # pink
  medium_color = '#77DD77'  # green
  high_color = '#AEC6CF'  # blue
  # low_color = '#FFE4E1'  # rose
  # medium_color = '#FFDAB9'  # peach
  # high_color = '#F08080'  # coral
  # low_color = '#008080'  # teal
  # medium_color = '#4169E1'  # royal blue
  # high_color = '#1E90FF'  # dodger blue
  overall_color = '#707070'  # grey

  x, y = zip(*rds)

  plt.scatter(x, y, alpha=0.1)
  plt.gca().set_xlim(0, 50)
  # plt.plot(x, y, color=overall_color, linewidth=2, linestyle='--')
  plt.xlabel('Games')
  plt.ylabel('Rating Deviation')

  plt.title('Increase of Glicko-2 Confidence Over Time')
  # plt.legend()
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  print(f'Read data from {path}...')
  rds = read_csv(path)
  plot(rds)
