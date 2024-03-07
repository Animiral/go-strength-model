from __future__ import annotations
import os
import csv
from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PlayerGameEntry:
    """Represents the data for one of the players in a game"""
    def __init__(self, name: str, rating: float, prevGame: Optional[GameEntry]):
        self.name = name
        self.rating = rating
        self.predictedRating = None
        self.prevGame = prevGame
        self.features = None
        self.recentMoves = None

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
    featureDims = 6  # TODO, adapt to whichever features we currently use

    def __init__(self, listpath: str, featuredir: str, marker: str):
        self.featuredir = featuredir
        self.players: Dict[str, GameEntry] = {}  # stores last occurrence of player
        self.games = List[GameEntry]

        with open(listpath, 'r') as listfile:
            reader = csv.DictReader(listfile)
            self.games = [self._makeGameEntry(r) for r in reader]

        self.marked = [g for g in self.games if g.marker == marker]

    def __len__(self):
        return len(self.marked)

    def __getitem__(self, idx):
        """Load recent move features on demand"""
        game = self.marked[idx]
        if game.black.recentMoves is None:
            self._fillRecentMoves(game.black.name, game)
        if game.white.recentMoves is None:
            self._fillRecentMoves(game.white.name, game)
        return game

    @staticmethod
    def _getScore(row):
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

    @staticmethod
    def _isSelected(self, row, setmarker):
        if "Set" in row.keys() and '*' != setmarker:
            return setmarker == row["Set"]
        else:
            return True

    def _makePlayerGameEntry(self, row, color):
        name = row['Player ' + color]
        rating = float(row[color + 'Rating'])
        prevGame = self.players.get(name, None)
        return PlayerGameEntry(name, rating, prevGame)

    def _makeGameEntry(self, row):
        sgfPath = row['File']
        black = self._makePlayerGameEntry(row, 'Black')
        white = self._makePlayerGameEntry(row, 'White')
        score = MovesDataset._getScore(row)
        marker = row["Set"]
        game = GameEntry(sgfPath, black, white, score, marker)
        self.players[black.name] = game  # set last occurrence
        self.players[white.name] = game  # set last occurrence
        return game

    def _loadFeatures(self, game: GameEntry):
        sgfPathWithoutExt, _ = os.path.splitext(game.sgfPath)
        game.black.features = MovesDataset._readFeaturesFromFile(f"{self.featuredir}/{sgfPathWithoutExt}_BlackFeatures.bin");
        game.white.features = MovesDataset._readFeaturesFromFile(f"{self.featuredir}/{sgfPathWithoutExt}_WhiteFeatures.bin");

    @staticmethod
    def _readFeaturesFromFile(path: str):
        FEATURE_HEADER = 0xfea70235  # feature file needs to start with this marker

        with open(path, 'rb') as file:
            # Read and validate the header
            header = np.fromfile(file, dtype=np.uint32, count=1)
            if header.size == 0 or header[0] != FEATURE_HEADER:
                raise IOError("Failed to read from feature file " + path)

            features_flat = np.fromfile(file, dtype=np.dtype(np.float32))

        count = len(features_flat) // MovesDataset.featureDims
        return torch.from_numpy(features_flat.reshape((MovesDataset.featureDims, count)))

    def _fillRecentMoves(self, player: str, game: GameEntry, window: int = 1000):
        recentMoves = torch.Tensor()
        count = 0
        gamePlayerEntry = game.playerEntry(player)
        historic = gamePlayerEntry.prevGame

        while count < window and historic is not None:
            sgfPathWithoutExt, _ = os.path.splitext(historic.sgfPath)
            entry = historic.playerEntry(player)
            if entry.features is None:
                color = 'Black' if historic.black.name == player else 'White'
                featurepath = f"{self.featuredir}/{sgfPathWithoutExt}_{color}Features.bin"
                entry.features = MovesDataset._readFeaturesFromFile(featurepath);
                recentMoves = torch.cat((entry.features, recentMoves), dim=1)
                count += entry.features.shape[1]

            # trim to window size if necessary
            if count > window:
                recentMoves = recentMoves[slice(None), slice(-window, None)]

            historic = entry.prevGame

        gamePlayerEntry.recentMoves = recentMoves

def collate_fn(batch):
    # padding?
    return batch

if __name__ == "__main__":
    print("Test moves_dataset.py")
    listpath = 'csv/games_labels.csv'
    featuredir = 'featurecache'
    marker = 'V'
    dataset = MovesDataset(listpath, featuredir, marker)
    print(f"Loaded {len(dataset.games)} games, {len(dataset)} of type {marker}.")

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    # loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    for game in loader:
        g = game[0]
        brecent = g.black.recentMoves.shape[1] if g.black.recentMoves.nelement() > 0 else 0
        wrecent = g.white.recentMoves.shape[1] if g.white.recentMoves.nelement() > 0 else 0
        print(f"{g.black.name} vs {g.white.name}: {brecent}/{wrecent} recent moves.")
