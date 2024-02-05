"""
Label each game in the provided list by the players' future rating to predict.
"""

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
	i = p.history.index(game)
	i = min(i+advance-1, len(p.history)-1)
	return p.rating[p.history[i]]

def main(listpath, outputpath, advance):
	# Input CSV format (title row):
	# File,Player White,Player Black,Winner,Judgement,WhiteWinrate,BlackRating,BlackDeviation,BlackVolatility,WhiteRating,WhiteDeviation,WhiteVolatility
	with open(listpath, "r") as listfile:
		reader = csv.DictReader(listfile)
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

	# second pass: lookup rating info for labels
	for row in rows:
		game = row["File"]
		white = row["Player White"]
		row["WhiteLabel"] = get_label(white, game, advance)
		black = row["Player Black"]
		row["BlackLabel"] = get_label(black, game, advance)

	# write output CSV file
	with open(outputpath, 'w') as outfile:
		fieldnames = ["File", "Player White", "Player Black", "WhiteLabel", "BlackLabel"]
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			slimrow = {k: row[k] for k in fieldnames}
			writer.writerow(slimrow)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label each game in the provided list by the players' future rating to predict.")
    parser.add_argument('--list', type=str, required=True, help='Path to the CSV file listing the SGF files and players ratings.')
    parser.add_argument('--output', type=str, required=True, help='Name of the CSV output file. (overwrites!)')
    parser.add_argument('--advance', type=int, required=True, default=10, help='Number of games to look into the future for label')
    args = parser.parse_args()

    main(args.list, args.output, args.advance)
