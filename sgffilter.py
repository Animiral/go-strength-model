#!/usr/bin/env python3

import os
import sys
import csv
import re
from sgfmill import sgf
from collections import namedtuple

canadian_pattern = re.compile(r'(\d+)/(\d+) Canadian')
fischer_pattern = re.compile(r'(\d+) fischer')
byoyomi_pattern = re.compile(r'(\d+)x(\d+) byo-yomi')
simple_pattern = re.compile(r'(\d+) simple')
valid_result_pattern = re.compile(r'[wWbB]\+[TR\d]')

def sec_per_move(overtime):
    """Determine seconds per move from the overtime specification."""

    match = canadian_pattern.search(overtime)
    if match:
        stones, time = map(int, match.groups())
        return time/stones

    match = fischer_pattern.search(overtime)
    if match:
        time_increment = int(match.group(1))
        return time_increment

    match = byoyomi_pattern.search(overtime)
    if match:
        periods, period_time = map(int, match.groups())
        return period_time

    match = simple_pattern.search(overtime)
    if match:
        time = int(match.group(1))
        return time

    return None

SgfProperties = namedtuple('SgfProperties', ['player_white', 'player_black', 'result', 'date', 'size', 'handicap', 'komi', 'time', 'overtime', 'comment', 'moves'])

def parse_sgf_properties(path):
    """Parse the SGF file and return basic properties."""
    with open(path, 'r') as f:
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
    result = get_or_default(root_node, 'RE', '')
    date = get_or_default(root_node, 'DT', '')
    size = root_node.get('SZ')
    handicap = get_or_default(root_node, 'HA', None)
    komi = root_node.get('KM')
    time = root_node.get('TM')
    overtime = get_or_default(root_node, 'OT', None)
    comment = get_or_default(root_node, 'GC', '')
    moves = len(game.get_main_sequence()) - 1  # move 0 is a node!
    return SgfProperties(player_white, player_black, result, date, size, handicap, komi, time, overtime, comment, moves)

def process_sgf(path, games_writer, player_files, error_files):
    """Determine info about the SGF found at the path and write the appropriate files."""
    properties = parse_sgf_properties(path)

    # FILTER: only allow 19x19
    if properties.size != 19:
        print(f"Blacklist {path}: not 19x19, but {properties.size}", file=sys.stderr)
        error_files["size"].write(f"{path},{properties.size}\n")
        return

    # FILTER: no handicap and proper komi
    #   examples:
    #   KM[-9]  - 9 komi for black
    if properties.handicap and properties.handicap != '0':
        print(f"Blacklist {path}: handicap {properties.handicap}", file=sys.stderr)
        error_files["handicap"].write(f"{path},H{properties.handicap}\n")
        return
    if properties.komi not in [6, 6.5, 7, 7.5]:
        print(f"Blacklist {path}: komi {properties.komi}", file=sys.stderr)
        error_files["handicap"].write(f"{path},{properties.komi}\n")
        return

    # FILTER: no blitz (<= 5 sec/move)
    #   examples:
    #   TM[5400]OT[25/600 Canadian]  - OTB tournament
    #   TM[259200]OT[86400 fischer]  - correspondence
    #   TM[600]OT[3x30 byo-yomi]     - live
    #   TM[0]OT[259200 simple]       - 3 days per move
    spm = sec_per_move(properties.overtime)
    if spm is None or spm <= 5:
        print(f"Blacklist {path}: sec_per_move {spm} (overtime {properties.overtime})", file=sys.stderr)
        error_files["time"].write(f"{path},{properties.overtime}\n")
        return

    # FILTER: short games (<20 moves)
    if properties.moves < 20:
        print(f"Blacklist {path}: {properties.moves} moves", file=sys.stderr)
        error_files["length"].write(f"{path},{properties.moves}\n")
        return

    # TODO FILTER: move 2 must not be PASS (improperly specified handicap game records)

    # FILTER: only allow ranked games
    #   examples:
    #   GC[correspondence,unranked]  - comment: unranked game
    if 'unranked' in properties.comment or not 'ranked' in properties.comment:
        print(f"Blacklist {path}: not ranked, comment: {properties.comment}", file=sys.stderr)
        error_files["ranked"].write(f"{path},{properties.comment}\n")
        return

    # FILTER: only allow results of counting, resignation, timeout
    #   examples:
    #   RE[B+F]   - black wins by forfeit
    #   RE[W+T]   - white wins by time
    match = valid_result_pattern.search(properties.result)
    if not match:
        print(f"Blacklist {path}: result {properties.result}", file=sys.stderr)
        error_files["result"].write(f"{path},{properties.result}\n")
        return

    games_writer.writerow([path, properties.player_white, properties.player_black, properties.result])
    for playername in [properties.player_white, properties.player_black]:
        if playername not in player_files:
            player_files[playername] = open(f"player/{playername}.txt", 'w')
        player_files[playername].write(path + "\n")

def main(directories, output_file='games.csv'):
    player_files = {}  # file handles to games list for every player
    error_files = {}  # file handles to filtered-out lists

    if not os.path.isdir("player"):
        os.mkdir("player")  # directory for files by player name

    try:
        error_files = {
            "size": open("fail_size.csv", 'w'),
            "handicap": open("fail_handicap.csv", 'w'),
            "time": open("fail_time.csv", 'w'),
            "length": open("fail_length.csv", 'w'),
            "ranked": open("fail_ranked.csv", 'w'),
            "result": open("fail_result.csv", 'w')
        }
        with open(output_file, 'w', newline='') as games_file:
            writer = csv.writer(games_file)
            writer.writerow(["File", "Player White", "Player Black", "Winner"])

            for directory in directories:
                for root, dirs, files in os.walk(directory):
                    files.sort() # naming convention must establish order in time

                    for filename in files:
                        if filename.endswith('.sgf'):
                            path = os.path.join(root, filename)
                            process_sgf(path, writer, player_files, error_files)

        # (maybe later FILTER: players with 10 or fewer games)

    finally:
        # Cleanup
        for handle in player_files.values():
            handle.close()
        for handle in error_files.values():
            handle.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse SGF files and write basic properties to a CSV.")
    parser.add_argument('directories', nargs="+", help='Path to the directories containing the SGF files.')
    parser.add_argument('--output', default='games.csv', help='Name of the CSV output file (default: games.csv).')
    args = parser.parse_args()

    main(args.directories, args.output)
