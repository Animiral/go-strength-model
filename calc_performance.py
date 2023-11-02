#!/usr/bin/env python3
# Calculate the prediction success rate and log-likelihood of a rating system run record CSV.

import csv
import math

def main(listpath):
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
        reader = csv.DictReader(infile)
        winner_title = "Winner"
        if "Judgement" in reader.fieldnames:
            winner_title = "Judgement"  # override column for result given by SGF

        for row in reader:
            player_white = row['Player White']
            player_black = row['Player Black']
            winner = row[winner_title].lower()
            winner_black = 'b' == winner[0]
            winner_white = 'w' == winner[0]
            winrate = float(row['WhiteWinrate'])
            if math.isnan(winrate):
                winrate = 0.5
            row_success = (winrate > 0.50001 and winner_white) or (winrate < 0.49999 and winner_black)
            row_withinfo = player_white in players or player_black in players
            row_fullinfo = player_white in players and player_black in players
            row_logp = float(winner_black) * math.log(1.-winrate) + float(winner_white) * math.log(winrate)

            count = count + 1
            success = success + int(row_success)
            zeroinfo = zeroinfo + int(not row_withinfo)
            oneinfo = oneinfo + int(row_withinfo and not row_fullinfo)
            success_withinfo = success_withinfo + int(row_success and row_withinfo)
            success_fullinfo = success_fullinfo + int(row_success and row_fullinfo)
            noresult = noresult + int(not winner_black and not winner_white)
            if winner_black or winner_white:
                logp = logp + row_logp
                if row_withinfo:
                    logp_withinfo = logp_withinfo + row_logp
                if row_fullinfo:
                    logp_fullinfo = logp_fullinfo + row_logp

            players.add(player_white)
            players.add(player_black)

    print(f"Finished counting run of {count} matchups between {len(players)} players.")
    print(f"Prediction accuracy: {success}/{count} ({success/count:.3f}), logp: {logp}")
    print(f"Without zero-info matchups: {success_withinfo}/{count-zeroinfo} ({success_withinfo/(count-zeroinfo):.3f}), logp: {logp_withinfo}")
    print(f"Only both-rated matchups: {success_fullinfo}/{count-zeroinfo-oneinfo} ({success_fullinfo/(count-zeroinfo-oneinfo):.3f}), logp: {logp_fullinfo}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate the prediction success rate and log-likelihood of a rating system run record CSV.")
    parser.add_argument("list", type=str, help='Path to the CSV file listing the games, results and winrate.')
    args = parser.parse_args()

    main(args.list)

