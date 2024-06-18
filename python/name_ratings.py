# Merge two instances of the same games list, one from `sgffilter.py` and one from goratings `analyze_glicko2_one_game_at_a_time.py`.
# The merged output list simply restores the paths to the SGFs and the player names from the original.
# The number and order of rows between the two inputs must match.
import csv

def main(listpath, ratingspath, outputpath):
    listfile = open(listpath, "r")
    listreader = csv.DictReader(listfile)
    ratingsfile = open(ratingspath, "r")
    ratingsreader = csv.DictReader(ratingsfile)

    fieldnames = listreader.fieldnames  # expect: ['File', 'Player White', 'Player Black', 'Score']
    # take these additional columns from the ratings file
    ratingsfields = ['PredictedScore','PredictedBlackRating','PredictedWhiteRating',
                     'BlackRating','BlackDeviation','BlackVolatility','WhiteRating','WhiteDeviation','WhiteVolatility']
    fieldnames += ratingsfields
    outputfile = open(outputpath, "w")
    writer = csv.DictWriter(outputfile, fieldnames)
    writer.writeheader()

    for listrow, ratingsrow in zip(listreader, ratingsreader):
        for field in ratingsfields:
            listrow[field] = ratingsrow[field]
        writer.writerow(listrow)

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

