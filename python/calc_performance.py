#!/usr/bin/env python3
# Calculate the prediction success rate and log-likelihood of a rating system run record CSV.

import csv
import math

# Glicko-2 expected score from OGS/goratings repository; without RD/volatility
def glickoScore(black_rating, white_rating) -> float:
    GLICKO2_SCALE = 173.7178
    return 1 / (1 + math.exp((white_rating - black_rating) / GLICKO2_SCALE))

def getScore(row):
    if "Score" in row.keys():
        return float(row["Score"])
    elif "Winner" in row.keys():
        winner = row["Winner"]
    elif "Judgement" in row.keys():
        winner = row["Judgement"]

    w = winner[0].lower()
    if 'b' == w:
        return 1
    elif 'w' == w:
        return 0
    else:
        print(f"Warning! Undecided game in dataset: {row['File']}")
        return 0.5  # Jigo and undecided cases

def getPredScore(row):
    cols = row.keys()  # available columns
    if "PredictedScore" in cols:
        ps = float(row["PredictedScore"])
    elif "WhiteWinrate" in cols:
        ps = 1 - float(row["WhiteWinrate"])
    elif ("BlackRating" in cols or "PredictedBlackRating" in cols) and ("WhiteRating" in cols or "PredictedWhiteRating" in cols):
        # use glicko score from ratings
        if "BlackRating" in cols:
            black_rating = float(row["BlackRating"])
        else:
            black_rating = float(row["PredictedBlackRating"])
        if "WhiteRating" in cols:
            white_rating = float(row["WhiteRating"])
        else:
            white_rating = float(row["PredictedWhiteRating"])
        ps = glickoScore(black_rating, white_rating)
    else:
        raise ValueError("Missing column for score prediction! (PredictedScore or WhiteWinrate or BlackRating+WhiteRating or PredictedBlackRating+PredictedWhiteRating)")

    if math.isnan(ps):
        ps = 0.5

    return ps

def isSelected(row, setmarker):
    if "Set" in row.keys() and '*' != setmarker:
        return setmarker == row["Set"]
    else:
        return True

def tolerant_log(x): # accepts x=0 and outputs -inf instead of ValueError
    if 0 == x:
        return -math.inf
    else:
        return math.log(x)

# Calculate rating performance on the list file, filter by the given set marker.
# If fixed_prediction is True, the result will be the same as predicting 50:50 every game.
def main(listpath, setmarker='V', fixed_prediction=False, novice_check=False):
    count = 0         # total records
    success = 0       # correctly predicted game result
    zeroinfo = 0      # number of records with first occurrence of both players
    oneinfo = 0       # number of records with first occurrence of one player
    success_withinfo = 0  # correct predictions with at least one known player
    success_fullinfo = 0  # correct predictions with both players known
    noresult = 0      # number of records where neither black nor white is winner (success impossible)
    logp = 0          # sum of log-likelihood of all predictions except noresult
    logp_withinfo = 0 # sum of log-likelihood of all predictions except noresult and zeroinfo
    logp_fullinfo = 0 # sum of log-likelihood of all predictions except noresult, zeroinfo and oneinfo
    players = set()

    # Input CSV format (title row):
    # File,Player White,Player Black,Winner,WhiteWinrate,BlackRating,WhiteRating
    with open(listpath, 'r') as infile:
        for row in csv.DictReader(infile):
            # keep track of players occurred, even for rows not selected
            player_white = row['Player White']
            player_black = row['Player Black']
            row_withinfo = player_white in players or player_black in players
            row_fullinfo = player_white in players and player_black in players
            players.add(player_white)
            players.add(player_black)

            if not isSelected(row, setmarker):
                continue

            score = getScore(row)
            predScore = 0.5 if fixed_prediction else getPredScore(row)
            if score > 0.5: # black win
                row_success = predScore > 0.5
            else: # white win
                row_success = predScore <= 0.5 # a dead center pred. counts as white due to a priori chance
            row_logp = tolerant_log(1.-abs(score-predScore))

            count = count + 1
            success = success + int(row_success)
            zeroinfo = zeroinfo + int(not row_withinfo)
            oneinfo = oneinfo + int(row_withinfo and not row_fullinfo)
            success_withinfo = success_withinfo + int(row_success and row_withinfo)
            success_fullinfo = success_fullinfo + int(row_success and row_fullinfo)
            # noresult = noresult + int(not winner_black and not winner_white)
            # if winner_black or winner_white:
            logp = logp + row_logp
            if row_withinfo:
                logp_withinfo = logp_withinfo + row_logp
            if row_fullinfo:
                logp_fullinfo = logp_fullinfo + row_logp

            # print(f"{player_black} vs {player_white}: s={score}, p={predScore}, log(p)={row_logp}")

    count_withinfo = count - zeroinfo
    count_fullinfo = count - zeroinfo - oneinfo
    logp /= count
    logp_withinfo /= count_withinfo
    logp_fullinfo /= count_fullinfo

    print(f"Finished counting run of {count} matchups between {len(players)} players.")
    print(f"Prediction accuracy: {success}/{count} ({success/count:.3f}), logp: {logp}")
    if novice_check:
        print(f"Without zero-info matchups: {success_withinfo}/{count_withinfo} ({success_withinfo/(count_withinfo):.3f}), logp: {logp_withinfo}")
        print(f"Only both-rated matchups: {success_fullinfo}/{count_fullinfo} ({success_fullinfo/count_fullinfo:.3f}), logp: {logp_fullinfo}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate the prediction success rate and log-likelihood of a rating system run record CSV.")
    parser.add_argument("list", type=str, help='Path to the CSV file listing the games, results and winrate.')
    parser.add_argument("-m", "--setmarker", type=str, default="*", help='Calculate on "T": training set, "V": validation set, "E": test set, "*": all')
    parser.add_argument("--fixed-prediction", action="store_true", help='Ignore predictions, instead predict 50:50 chances on every single game')
    parser.add_argument("-n", "--novice-check", action="store_true", help='Print extra stats based on player\'s first occurrences')
    args = parser.parse_args()

    if args.setmarker not in ['T', 'V', 'E', '*']:
        raise ValueError("Set marker must be one of 'T', 'V', 'E', '*'.")

    main(args.list, args.setmarker, args.fixed_prediction, args.novice_check)

