#!/usr/bin/env python3
"""
Usage: ogsratings.py INPUTFILE OUTPUTFILE

Given a CSV file containing game IDs, send a request to OGS for each game for the rating of the players
in the game at the time.
Store the output in a CSV file.
"""

import sys
import requests
import csv
import re
import time

# Note: this is how to get info from OGS API
# curl https://online-go.com/termination-api/player/1350141/v5-rating-history
# curl -X 'GET' 'https://online-go.com/api/v1/games/65215166' -H 'accept: application/json'
# documentation at https://ogs.docs.apiary.io

def main(inputpath, outputpath):
  pattern = re.compile(r'\/(\d+)-')  # matches game id at beginning of file name, after dir
  with open(inputpath, "r") as infile:
    reader = csv.DictReader(infile)
    with open(outputpath, "w") as outfile:
      fieldnames = ["File","BlackRating","WhiteRating","PredictedBlackRating","PredictedWhiteRating"]
      writer = csv.DictWriter(outfile, fieldnames=fieldnames)
      writer.writeheader()
      for row in reader:
        filename = row["File"]
        gameid = pattern.search(filename).group(1)
        url = f"https://online-go.com/api/v1/games/{gameid}"
        try:
          response = requests.get(url)
          response_json = response.json()
          row["BlackRating"] = response_json["historical_ratings"]["black"]["ratings"]["overall"]["rating"]
          row["WhiteRating"] = response_json["historical_ratings"]["white"]["ratings"]["overall"]["rating"]
          print(filename, row["BlackRating"], row["WhiteRating"])
          writer.writerow(row)
        except Exception as e:
          print(e)  # just go on with the loop even if some reqs fail

        time.sleep(1)  # do not flood the server

if __name__ == "__main__":
  """
  Given a CSV file containing game IDs, send a request to OGS for each game for the rating of the players in the game at the time.
  Store the output in a CSV file.
  """
  inputpath = sys.argv[1]
  outputpath = sys.argv[2]
  print(f"Read game IDs from {inputpath}, write results to {outputpath}.")
  main(inputpath, outputpath)
