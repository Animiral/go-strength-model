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

def plot_ratings(x, y, modelname, setname):
  plt.scatter(x, y, alpha=0.1)
  plt.xlabel("Rating")
  plt.ylabel("Model")
  plt.title(f"{modelname} vs Labels in {setname}")
  plt.show()

def plot_score(whitewins, blackwins, modelname, setname):
  colors_left = plt.cm.RdYlGn_r(whitewins)
  colors_right = plt.cm.RdYlGn(blackwins)
  fig, (left, right) = plt.subplots(1, 2, figsize=(6, 4))

  left.set_facecolor("beige")
  left.scatter(range(len(whitewins)), whitewins, color=colors_left, label="White Wins", alpha=0.1)
  left.set_xticks([])
  left.set_ylabel("Est.Score")
  left.set_title(f"White Wins", fontsize=12)

  right.set_facecolor("dimgray")
  right.scatter(range(len(blackwins)), blackwins, color=colors_right, label="Black Wins", alpha=0.1)
  right.set_xticks([])
  right.set_ylabel("Est.Score")
  right.set_title(f"Black Wins", fontsize=12)

  fig.suptitle(f"{modelname} vs Outcomes in {setname}")
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  path = sys.argv[1]
  modelname = sys.argv[2] if len(sys.argv) > 2 else "Model"

  print(f"Read data from {path}...")
  ratings, scores = read_csv(path)
  for setname, ratlist in ratings.items():
    setname = {"T": "Training Set", "V": "Validation Set", "E": "Test Set"}[setname]
    print(f"Preparing {setname} ratings...")
    x, y = zip(*ratlist)
    plot_ratings(x, y, modelname, setname)

  for setname, scolist in scores.items():
    setname = {"T": "Training Set", "V": "Validation Set", "E": "Test Set"}[setname]
    print(f"Preparing {setname} scores...")

    whitewins = sorted([s[1] for s in scolist if s[0] < 0.5])
    blackwins = sorted([s[1] for s in scolist if s[0] > 0.5])
    plot_score(whitewins, blackwins, modelname, setname)
