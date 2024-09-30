# Print statistics on judged games vs raw games.
import sys
import math
import csv
import os

def read_csv(path, score_column):
  score = {}
  with open(path, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
      score[row["File"]] = row[score_column]
  return score

def stats_games(games_path, judged_path):
  winners = read_csv(games_path, "Winner")
  scores = read_csv(judged_path, "Score")

  games_count = 0
  judged_count = 0
  black_flipped = 0  # black win changed to white
  white_flipped = 0  # white win changed to black
  counted_flipped = 0  # end result changed from counted board
  timeout_flipped = 0  # end result changed from timeout
  resign_flipped = 0  # end result changed from resignation

  for path, winner in winners.items():
    games_count += 1
    score = scores.get(path)
    if score is None:
      continue
    judged_count += 1

    flipped = False

    if winner.startswith("B") and "0" == score:
      black_flipped += 1
      flipped = True
    if winner.startswith("W") and "1" == score:
      white_flipped += 1
      flipped = True

    if flipped:
      if winner[-1].isdigit():
        counted_flipped += 1
      if winner.endswith("T"):
        timeout_flipped += 1
      if winner.endswith("R"):
        resign_flipped += 1

  return games_count, judged_count, black_flipped, white_flipped, counted_flipped, timeout_flipped, resign_flipped

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Expected args: games.csv games_judged.csv")
    sys.exit(1)

  games_path = sys.argv[1]
  judged_path = sys.argv[2]
  if not os.path.isfile(games_path) or not os.path.isfile(judged_path):
    print("Expected args (must be files): games.csv games_judged.csv")
    sys.exit(1)

  print("Read games list from " + games_path)
  print("Read judged games list from " + judged_path)

  games_count, judged_count, black_flipped, white_flipped, counted_flipped, timeout_flipped, resign_flipped = stats_games(games_path, judged_path)
  total_flipped = black_flipped + white_flipped

  print(f"Games: {games_count} total ({judged_count}/{100*judged_count/games_count:.2f}% judged, {games_count-judged_count}/{100*(games_count-judged_count)/games_count:.2f}% omitted)")
  print(f"Flipped: {total_flipped} ({100*total_flipped/judged_count:.2f}%)")
  print(f"  black-to-white {black_flipped} ({100*black_flipped/total_flipped:.2f}%)")
  print(f"  white-to-black {white_flipped} ({100*white_flipped/total_flipped:.2f}%)")
  print(f"  after counting {counted_flipped} ({100*counted_flipped/total_flipped:.2f}%)")
  print(f"  after timeout {timeout_flipped} ({100*timeout_flipped/total_flipped:.2f}%)")
  print(f"  after resignation {resign_flipped} ({100*resign_flipped/total_flipped:.2f}%)")
