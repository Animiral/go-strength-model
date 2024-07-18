#!/usr/bin/env python3
# Usage: python/more/age.py --list csv/games_glicko.csv --output csv/age.csv --advance 10
# 1. determine age of players in the dataset (how many games they have played to that point)
# 2. determine age of label (how many games before that label)

import argparse
import csv
import io

class Player:
    rating: dict[str, int]   # by game
    history: list[str]       # games in order

    def __init__(self):
        self.rating = {}
        self.history = []

players = {}  # every player's game history, lazy read

def store_rating(player, game, rating):
	if not player in players:
		players[player] = Player()

	players[player].rating[game] = rating
	players[player].history.append(game)

def get_label(player, game, advance):
	p = players[player]
	age = p.history.index(game)
	labelage = min(age+advance-1, len(p.history)-1)
	return p.rating[p.history[labelage]], age+1, labelage+1

def main(listpath, outputpath, advance):
	# Required input CSV columns (title row):
	#   File,Player White,Player Black,BlackRating,WhiteRating
	# Optional rows copied:
	#   Winner,Judgement,Score,Set
	with open(listpath, "r") as listfile:
		reader = csv.DictReader(listfile)
		inputfields = reader.fieldnames
		rows = list(reader)

    # first pass: extract rating info
	for row in rows:
		game = row["File"]
		white = row["Player White"]
		white_rating = row["WhiteRating"]
		store_rating(white, game, white_rating)
		black = row["Player Black"]
		black_rating = row["BlackRating"]
		store_rating(black, game, black_rating)

	outrows = []

	# second pass: lookup rating and age info
	for row in rows:
		game = row["File"]
		white = row["Player White"]
		rating, age, _ = get_label(white, game, advance)
		outrows.append({"Rating": rating, "Age": age})
		black = row["Player Black"]
		rating, age, _ = get_label(black, game, advance)
		outrows.append({"Rating": rating, "Age": age})

	# write output CSV file
	with open(outputpath, 'w') as outfile:
		fieldnames = ["Rating", "Age"]
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in outrows:
			slimrow = {k: row[k] for k in fieldnames}
			writer.writerow(slimrow)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a list of labels (player ratings) and how many games worth of info they contain.")
    parser.add_argument('--list', type=str, required=True, help='Path to the CSV file listing the SGF files and players ratings.')
    parser.add_argument('--output', type=str, required=True, help='Name of the CSV output file. (overwrites!)')
    parser.add_argument('--advance', type=int, required=True, default=10, help='Number of games to look into the future for label')
    args = parser.parse_args()

    main(args.list, args.output, args.advance)
