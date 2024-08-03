#!/usr/bin/env python3
# Usage: plots/estimate_vs_label.py csv/games_glicko.csv Glicko-2
"""
Given an input CSV file describing predicted/actual ratings (and score),
plot them vs the labels from the same file. Optional set marker (default E)
"""
import argparse
import sys
import csv
import matplotlib.pyplot as plt
import fontconfig

def read_csv(path):
  global max_age_dev
  ratings = {}  # list of (label, estimate) for every set marker
  scores = {}  # list of (label, estimate) for every set marker

  with open(path, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
      setname = row["Set"]
      if "-" == setname:
        continue  # not in any set

      ratlist = ratings.setdefault(setname, [])
      white_rating = float(row["WhiteRating"])
      predicted_white_rating = float(row["PredictedWhiteRating"])
      ratlist.append((white_rating, predicted_white_rating))
      black_rating = float(row["BlackRating"])
      predicted_black_rating = float(row["PredictedBlackRating"])
      ratlist.append((black_rating, predicted_black_rating))

      scolist = scores.setdefault(setname, [])
      score = float(row["Score"])
      predicted_score = float(row["PredictedScore"])
      scolist.append((score, predicted_score))

  return ratings, scores

def setup_ratings(ax, modelname: str = "Model", setname: str = "Set"):
  ax.set_xlabel("Rating")
  ax.set_ylabel("Model")
  ax.set_title(f"{modelname} vs Labels in {setname}")

def plot_ratings(ax, x, y):
  minx = min(x) - 50
  maxx = max(x) + 50
  ax.set_xlim(minx, maxx)
  ax.set_ylim(minx, maxx)
  ax.scatter(x, y, alpha=0.1)
  ax.plot([minx, maxx], [minx, maxx], linestyle="--", color="tab:cyan")

def setup_score(ax_white, ax_black):
  ax_white.set_facecolor("beige")
  ax_white.set_xticks([])
  ax_white.set_ylabel("Est.Score")
  ax_white.set_title(f"White Wins", fontsize=12)

  ax_black.set_facecolor("dimgray")
  ax_black.set_xticks([])
  ax_black.set_ylabel("Est.Score")
  ax_black.set_title(f"Black Wins", fontsize=12)

def plot_score(ax_white, ax_black, whitewins, blackwins):
  colors_white = plt.cm.RdYlGn_r(whitewins)
  colors_black = plt.cm.RdYlGn(blackwins)
  ax_white.scatter(range(len(whitewins)), whitewins, color=colors_white, label="White Wins", alpha=0.1)
  ax_black.scatter(range(len(blackwins)), blackwins, color=colors_black, label="Black Wins", alpha=0.1)

if __name__ == "__main__":
  description = """
  Given an input CSV file describing predicted/actual ratings (and score),
  plot them vs the labels from the same file. Optional set marker (default E)
  """
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("-m", "--setmarker", type=str, default="E", help="Marker of the set to plot (T/V/E/X).")
  parser.add_argument("-s", "--scoredist", action="store_true", help="Plot the score distribution.")
  parser.add_argument("path", type=str, help="Path to the dataset list CSV file.")
  parser.add_argument("title", type=str, default="Model", help="Name of the evaluated function for plot title.")
  args = parser.parse_args()
  print(vars(args))

  # fig, axs = plt.subplots(2, 2, figsize=(6, 6))  # all in one
  # ax_r = axs[0, 0]
  # ax_w = axs[1, 0]
  # ax_b = axs[1, 1]

  ratings, scores = read_csv(args.path)
  setname = {"T": "Training Set", "V": "Validation Set", "E": "Test Set"}[args.setmarker]

  if args.scoredist:
    fig, axs = plt.subplots(1, 2, figsize=(6, 4))  # two
    ax_w = axs[0]
    ax_b = axs[1]
    scores = scores[args.setmarker]
    print(f"Preparing {setname} scores...")

    setup_score(ax_w, ax_b)
    whitewins = sorted([s[1] for s in scores if s[0] < 0.5])
    blackwins = sorted([s[1] for s in scores if s[0] > 0.5])
    plot_score(ax_w, ax_b, whitewins, blackwins)
    plt.tight_layout()
  else:
    fig, axs = plt.subplots(figsize=(6, 4))  # just one
    ax_r = axs
    ratings = ratings[args.setmarker]
    print(f"Preparing {setname} ratings...")

    setup_ratings(ax_r, args.title, setname)
    x, y = zip(*ratings)
    plot_ratings(ax_r, x, y)

  plt.show()
