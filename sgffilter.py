#!/usr/bin/env python3

import os
import csv
from sgfmill import sgf

def parse_sgf_properties(filepath):
    """Parse the SGF file and return basic properties."""
    with open(filepath, 'r') as f:
        sgf_data = f.read()
    game = sgf.Sgf_game.from_string(sgf_data)

    def get_or_default(sgf_node, identifier, default):
        if sgf_node.has_property(identifier):
            return sgf_node.get(identifier)
        else:
            return default

    root_node = game.get_root()
    player_white = root_node.get('PW')
    player_black = root_node.get('PB')
    winner = get_or_default(root_node, 'RE', '')
    date = get_or_default(root_node, 'DT', '')
    return player_white, player_black, winner, date

def main(directory_path, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["File", "Player White", "Player Black", "Winner", "Date"])

        for root, dirs, files in os.walk(directory_path):
            # Sort directories and files for alphabetic traversal
            dirs.sort()
            files.sort()

            for filename in files:
                if filename.endswith('.sgf'):
                    filepath = os.path.join(root, filename)
                    player_white, player_black, winner, date = parse_sgf_properties(filepath)
                    # FILTER: only allow 19x19
                    # FILTER: no handicap
                    # FILTER: no blitz (<= 5 sec/move)
                    # FILTER: short games (<20 moves)
                    # FILTER: only allow ranked games
                    # FILTER: only allow results of counting, resignation, timeout
                    writer.writerow([filepath, player_white, player_black, winner, date])

    # FILTER: players with 10 or fewer games

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse SGF files and write basic properties to a CSV.")
    parser.add_argument('directory', help='Path to the directory containing the SGF files.')
    parser.add_argument('--output', default='output.csv', help='Name of the CSV output file (default: output.csv).')
    args = parser.parse_args()

    main(args.directory, args.output)
