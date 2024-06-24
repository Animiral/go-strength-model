from __future__ import annotations
import os
from os.path import exists
import csv
import re
from typing import List, Optional
import zipfile
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PlayerGameEntry:
    """Represents the data for one of the players in a game"""
    def __init__(self, name: str, rating: float, prevGame: Optional[GameEntry]):
        self.name = name
        self.rating = rating
        self.predictedRating = None
        self.prevGame = prevGame

class GameEntry:
    """Represents one game in the list"""
    def __init__(self, sgfPath: str, black: PlayerGameEntry, white: PlayerGameEntry, score: float, marker: str):
        self.sgfPath = sgfPath
        self.black = black
        self.white = white
        self.score = score
        self.predictedScore = None
        self.marker = marker

    def playerEntry(self, name: str):
        if self.black.name == name:
            return self.black
        elif self.white.name == name:
            return self.white
        else:
            raise Exception(f"Player {name} does not occur in game {self.sgfPath}.")

class MovesDataset(Dataset):
    """Load the dataset from a CSV list file"""

    def __init__(self, listpath: str, featuredir: str, marker: str, *,
      featurename: str = "pick", sparse: bool = True):
        self.featuredir = featuredir
        self.players: Dict[str, GameEntry] = {}  # stores last occurrence of player
        self.games = List[GameEntry]

        with open(listpath, "r") as listfile:
            reader = csv.DictReader(listfile)
            self.games = [self._makeGameEntry(r) for r in reader if (not sparse) or marker == r["Set"]]

        self.marked = [g for g in self.games if g.marker == marker]
        self.featurename = featurename  # used to select correct feature data from ZIP
        self.featureDims = self._findFeatureDims()

    def __len__(self):
        return len(self.marked)

    def __getitem__(self, idx):
        """Load recent move features from disk"""
        game = self.marked[idx]
        blackRecent = self.loadRecentMoves("Black", game)
        whiteRecent = self.loadRecentMoves("White", game)
        return (blackRecent, whiteRecent, game.black.rating, game.white.rating, game.score)

    def write(self, outpath: str):
        """Write to CSV file including predictions data where applicable"""
        with open(outpath, "w") as outfile:
            fieldnames = ["File","Player White","Player Black","Score","BlackRating","WhiteRating","PredictedScore","PredictedBlackRating","PredictedWhiteRating","Set"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for game in self.marked:
                row = {
                    "File": game.sgfPath,
                    "Player White": game.white.name,
                    "Player Black": game.black.name,
                    "Score": game.score,
                    "BlackRating": game.black.rating,
                    "WhiteRating": game.white.rating,
                    "PredictedScore": game.predictedScore,
                    "PredictedBlackRating": game.black.predictedRating,
                    "PredictedWhiteRating": game.white.predictedRating,
                    "Set": game.marker
                }
                writer.writerow(row)

    def loadRecentMoves(self, player: str, game: GameEntry, featureName: str = ""):
        assert player in {"Black", "White"}
        if "" == featureName:
            featureName = self.featureName  # default
            featureDims = self.featureDims  # known/detected by _findFeatureDims
        else:
            featureDims = -1  # guess

        basePath, _ = os.path.splitext(game.sgfPath)
        featurePath = f"{self.featuredir}/{basePath}_{player}Recent.zip"
        return load_features_from_zip(featurePath, featureName, featureDims)

    def _findFeatureDims(self):
        """Discover feature dimensions by loading recent move data, assuming they are consistent."""
        self.featureDims = -1  # this causes reshape() in loadRecentMoves to guess
        for game in self.games:
            try:
                data = self.loadRecentMoves('Black', game)
            except FileNotFoundError:
                continue
            if len(data) > 0:
                return data.shape[1]
        raise ValueError("Cannot discover feature dims: no recent move data found for any game.")

    @staticmethod
    def _getScore(row):
        if "Score" in row.keys():
            return float(row["Score"])
        elif "Winner" in row.keys():
            winner = row["Winner"]
        elif "Judgement" in row.keys():
            winner = row["Judgement"]

        w = winner[0].lower()
        if "b" == w:
            return 1
        elif "w" == w:
            return 0
        else:
            print(f"Warning! Undecided game in dataset: {row['File']}")
            return 0.5  # Jigo and undecided cases

    @staticmethod
    def _isSelected(self, row, setmarker):
        if "Set" in row.keys() and "*" != setmarker:
            return setmarker == row["Set"]
        else:
            return True

    def _makePlayerGameEntry(self, row, color):
        name = row["Player " + color]
        rating = float(row[color + "Rating"])
        prevGame = self.players.get(name, None)
        return PlayerGameEntry(name, rating, prevGame)

    def _makeGameEntry(self, row):
        sgfPath = row["File"]
        black = self._makePlayerGameEntry(row, "Black")
        white = self._makePlayerGameEntry(row, "White")
        score = MovesDataset._getScore(row)
        marker = row["Set"]
        game = GameEntry(sgfPath, black, white, score, marker)
        self.players[black.name] = game  # set last occurrence
        self.players[white.name] = game  # set last occurrence
        return game

def load_features_from_zip(path: str, featureName: str, featureDims: int = -1):
    with zipfile.ZipFile(path, "r") as z:
        with z.open("turn.bin") as file:
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            movecount = file_size // 4  # each turn index is a 32-bit int

        if 0 == movecount:
            return torch.empty(0, featureDims if featureDims > 0 else 0)

        with z.open(f"{featureName}.bin") as file:
            data = np.frombuffer(file.read(), dtype=np.float32)

    return torch.tensor(data).reshape(movecount, featureDims)

def pad_collate_one(rs):
    lens = [r.shape[0] for r in rs]
    rs = [r for r in rs if len(r) > 0]
    collated = torch.cat(rs, dim=0) if rs else torch.empty((0,0))
    return lens, collated

def pad_collate(batch):
    brecent, wrecent, brating, wrating, score = zip(*batch)
    blens, brecent = pad_collate_one(brecent)
    wlens, wrecent = pad_collate_one(wrecent)
    brating, wrating, score = map(torch.Tensor, (brating, wrating, score))
    return brecent, wrecent, blens, wlens, brating, wrating, score

class MovesDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["collate_fn"] = pad_collate
        super().__init__(*args, **kwargs)



# Test/debug code (run module as standalone)

def debugSummary(dataset):
    def vecChecksum(vec):
        # condense vec to a single printable number, likely different from vecs (trunks) of other positions,
        # but also close in value to very similar vecs (tolerant of float inaccuracies)
        accum = 0.0
        sos = 0.0
        weight = 1.0
        decay = 0.9999667797285222 # = pow(0.01, (1/(vec.size()-1))) -> smallest weight is 0.01

        for v in vec:
            accum += v * weight
            sos += v * v
            weight *= decay

        return accum + math.sqrt(sos)

    sgfPath = "dataset/2005/12/29/13067-NPC-Reepicheep.sgf"
    player = "White"
    gameEntry = next((ge for ge in dataset.games if ge.sgfPath == sgfPath), None)
    data = dataset.loadRecentMoves(player, gameEntry)
    print(f"Found {data.shape[0]} recent picks for {sgfPath}.")
    # ZIP_MOVEINDEX=132  # 123 + 9
    for row in data:
        print(f"pick {vecChecksum(row)}")

if __name__ == "__main__":
    print("Test moves_dataset.py")
    listpath = "csv7M/games_labels.csv"
    featuredir = "featurecache"
    marker = "V"
    dataset = MovesDataset(listpath, featuredir, marker, sparse=False)
    print(f"Loaded {len(dataset.games)} games, {len(dataset)} of type {marker}.")
    debugSummary(dataset)

    # loader = DataLoader(dataset, batch_size=3, collate_fn=pad_collate)

    # for bx, wx, blens, wlens, by, wy, score in loader:
    #     print(f"Got batch size bx: {bx.shape} ({blens}), wx: {wx.shape} ({wlens}); {by}; {wy}; {score}.")
