#!/usr/bin/env python3
# MovesDataset and MovesDataLoader allows the model to load train/val/test data.

from __future__ import annotations
import os
import csv
import re
import zipfile
import math
import numpy as np
import torch
from typing import Optional, List
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

    GLICKO2_SCALE = 173.7178   # from goratings Glicko-2 implementation
    GLICKO2_MEAN = 1623.1912875700598   # empirical mean of training labels
    GLICKO2_STDEV = 315.8087941393861   # empirical stdev of training labels

    def __init__(self, listpath: str, featuredir: str, marker: str, *,
      featurename: str = "pick", sparse: bool = True, featurememory: bool = True):
        """
        Load the dataset records from the CSV file at `listpath`.

        Args:
            listpath: Path to the file listing all the games
            featuredir: Directory where precomputed move features must be available
            marker: The set to operate on: "T": training, "V": validation, "E": test

        Keyword Args:
            featurename: Name of the feature set to use.
            sparse: Only keep rows that match the set marker.
            featurememory: If True, keep features in memory.
        """
        self.featuredir = featuredir
        self.players: Dict[str, GameEntry] = {}  # stores last occurrence of player

        with open(listpath, "r") as listfile:
            reader = csv.DictReader(listfile)
            self.games = [self._makeGameEntry(r) for r in reader if (not sparse) or marker == r["Set"]]

        self.marked = [g for g in self.games if g.marker == marker]
        self.features = {} if featurememory else None  # cache filled on demand
        self.featureName = featurename  # used to select correct feature data from ZIP
        self.featureDims = self._findFeatureDims()

    def __len__(self):
        return len(self.marked)

    def __getitem__(self, idx):
        """Load recent move features from disk"""
        game = self.marked[idx]
        blackRecent = self.loadRecentMoves("Black", game)
        whiteRecent = self.loadRecentMoves("White", game)
        # renorm labels into NN output range ~ N(0, 1)
        mu = MovesDataset.GLICKO2_MEAN
        scale = MovesDataset.GLICKO2_STDEV
        return (blackRecent, whiteRecent, (game.black.rating-mu)/scale, (game.white.rating-mu)/scale, game.score)

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

        # get features from memory cache (optional)
        key = (game.sgfPath, player, featureName)
        if self.features is not None and key in self.features:
            return self.features[key]

        # get features from uncompressed storage
        basePath, _ = os.path.splitext(game.sgfPath)
        path = f"{self.featuredir}/{basePath}_{player}Recent_{featureName}.npy"
        if os.path.exists(path):
            features = torch.from_numpy(np.load(path))
        else:
            # get features from compressed storage
            path = f"{self.featuredir}/{basePath}_{player}Recent.zip"
            features = load_features_from_zip(path, featureName, featureDims)

        # store features in memory cache (optional)
        if self.features is not None:
            self.features[key] = features

        return features

    def preload(self):
        """
        Load all marked game features of self.featureName into feature memory for faster access.
        """
        if self.features is not None:
            for game in self.marked:
                self.loadRecentMoves("Black", game)
                self.loadRecentMoves("White", game)

    def _findFeatureDims(self):
        """Discover feature dimensions by loading recent move data, assuming they are consistent."""
        self.featureDims = -1  # this causes reshape() in loadRecentMoves to guess
        for game in self.games:
            try:
                data = self.loadRecentMoves("Black", game)
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

class MovesDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["collate_fn"] = self.pad_collate
        super().__init__(*args, **kwargs)

    def pad_collate_one(self, rs):
        lens = [r.shape[0] for r in rs]               # get lengths
        rs = [r for r in rs if len(r) > 0]            # remove empties
        collated = torch.cat(rs, dim=0) if rs else torch.empty((0,0))
        return lens, collated

    def pad_collate(self, batch):
        brecent, wrecent, brating, wrating, score = zip(*batch)
        blens, brecent = self.pad_collate_one(brecent)
        wlens, wrecent = self.pad_collate_one(wrecent)
        brating, wrating, score = map(torch.Tensor, (brating, wrating, score))
        return brecent, wrecent, blens, wlens, brating, wrating, score

def bradley_terry_score(black_rating: float, white_rating: float) -> float:
    """Estimate the match score between two ratings determined by model output (same scale as labels)"""
    scale = MovesDataset.GLICKO2_STDEV
    return 1 / (1 + (10 ** ((white_rating - black_rating) * scale / 400)))

def scale_rating(rating):
    """Convert a rating from label scale to Glicko-2 scale"""
    return rating * MovesDataset.GLICKO2_STDEV + MovesDataset.GLICKO2_MEAN

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
