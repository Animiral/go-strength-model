#!/usr/bin/env python3
# Usage: plots/trickratings.py csv/trickratings.csv
"""
Given an input CSV file containing failure and refutation ratings for trick play problems,
plot them in a scatter plot to show their distribution.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import fontconfig
from rank import *

if __name__ == "__main__":
  description = """
  Given an input CSV file containing failure and refutation ratings for trick play problems,
  plot them in a scatter plot to show their distribution.
  """
  df = pd.read_csv(sys.argv[1])
  df = df.sort_values("Failure")
  fig, ax = plt.subplots(figsize=(5.6, 4))

  # lines
  for i in range(len(df)):
    plt.plot([df["Failure"].iloc[i], df["Refutation"].iloc[i]], [i, i], color="black", linewidth=0.5, zorder=-1)

  # points
  plt.scatter(df["Failure"], range(80), label="Failure")
  plt.scatter(df["Refutation"], range(80), label="Refutation")

  # mean indicator
  plt.axvline(df["Refutation"].mean(), color="orange", linestyle="--", linewidth=1, zorder=3)

  plt.legend()

  # axis labels
  ax.set_xlabel("Rating")
  plt.yticks([])  # rowcount is not important

  rank_axis = ax.twiny()
  # print(f"Set xlim to {ax.get_xlim()} -> {(rank(ax.get_xlim()[0]), rank(ax.get_xlim()[1]))}")
  # rank_ticks = rank(np.linspace(start=min(x), stop=max(x), num=10))
  rank_ticks = np.array([15, 20, 25, 30, 33, 36])
  rank_axis.set_xticks(to_rating(rank_ticks))
  ax.set_xlim(900, 2600)
  rank_axis.set_xlim(900, 2600)
  rank_axis.set_xticklabels(rankintstr(r) for r in rank_ticks)
  rank_axis.set_xlabel("Rank")

  plt.show()
