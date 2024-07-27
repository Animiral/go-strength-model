#!/usr/bin/env python3
# Usage: plots/estimate_vs_label.py csv/games_glicko.csv Glicko-2
"""
Given an input CSV file describing predicted/actual ratings (and score),
plot them vs the labels from the same file. One plot per set marker (T/V/E).
"""
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
  path = sys.argv[1]
  modelname = sys.argv[2] if len(sys.argv) > 2 else "Model"

  fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))

  # TODO: fix the loops to go back to having all figures
  print(f"Read data from {path}...")
  ratings, scores = read_csv(path)
  for setname, ratlist in ratings.items():
    setname = {"T": "Training Set", "V": "Validation Set", "E": "Test Set"}[setname]
    print(f"Preparing {setname} ratings...")

    setup_ratings(axs[0, 0], modelname, setname)
    x, y = zip(*ratlist)
    plot_ratings(axs[0, 0], x, y)

  for setname, scolist in scores.items():
    setname = {"T": "Training Set", "V": "Validation Set", "E": "Test Set"}[setname]
    print(f"Preparing {setname} scores...")

    whitewins = sorted([s[1] for s in scolist if s[0] < 0.5])
    blackwins = sorted([s[1] for s in scolist if s[0] > 0.5])
    plot_score(axs[1, 0], axs[1, 1], whitewins, blackwins)

  # plt.tight_layout()
  plt.show()
