# Merge two instances of the same games list, one from `sgffilter.py` and one from goratings `analyze_glicko2_one_game_at_a_time.py`.
# The merged output list simply restores the paths to the SGFs and the player names from the original.

def main(listpath, ratingspath, outputpath):
	listfile = open(listpath, "r")
	ratingsfile = open(ratingspath, "r")
	outputfile = open(outputpath, "w")

	while True:
		listline = listfile.readline().rstrip()
		ratingsline = ratingsfile.readline()
		if "" == listline:
			break  # EOF
		comma = ratingsline.find(',')           # first comma after game id
		comma = ratingsline.find(',', comma+1)  # second comma after black id
		comma = ratingsline.find(',', comma+1)  # third comma after white id
		outputline = listline + ratingsline[comma:]  # join all information
		outputfile.write(outputline)

	outputfile.close()
	listfile.close()
	ratingsfile.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge two instances of the same games list to restores SGF paths and player names.")
    parser.add_argument('--list', type=str, required=True, help='Path to the CSV file listing the SGF files and players.')
    parser.add_argument('--ratings', type=str, required=True, help='Path to the CSV file listing the same games plus rating information.')
    parser.add_argument('--output', type=str, required=True, help='Name of the CSV output file. (overwrites!)')
    args = parser.parse_args()

    main(args.list, args.ratings, args.output)

