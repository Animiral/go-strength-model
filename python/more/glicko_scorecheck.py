# Calculate expected score based on Glicko-2 ratings and compare with reference in one CSV
import sys
import math
import csv

inputpath = sys.argv[1]
samples = 100
print(f"Testing {samples} samples from {inputpath}...")

GLICKO2_SCALE = 173.7178

class Glicko2Entry:
    rating: float
    deviation: float
    volatility: float
    mu: float
    phi: float

    def __init__(self, rating: float = 1500, deviation: float = 350, volatility: float = 0.06) -> None:
        self.rating = rating
        self.deviation = deviation
        self.volatility = volatility
        self.mu = (self.rating - 1500) / GLICKO2_SCALE
        self.phi = self.deviation / GLICKO2_SCALE

entries = dict()  # name -> entry

def expected_win_probability(black: Glicko2Entry, white: Glicko2Entry) -> float:
    # See "Expected outcome of a game" in http://www.glicko.net/glicko/glicko.pdf
    # and formula for g(phi) in http://www.glicko.net/glicko/glicko2.pdf
    g = 1 / math.sqrt(1 + 3 * (black.phi**2 + white.phi**2) / math.pi**2)
    E = 1 / (
        1 + (
            math.exp(-g * (black.mu - white.mu))
        )
    )
    return E

with open(inputpath, "r") as inputfile:
	reader = csv.DictReader(inputfile)
	mse = 0
	line = 0
	for row in reader:
		blackname = row["Player Black"]
		whitename = row["Player White"]
		black = entries.get(blackname, Glicko2Entry())
		white = entries.get(whitename, Glicko2Entry())
		expected = float(row["PredictedScore"])

		actual = expected_win_probability(black, white)
		mse += (expected - actual)**2

		entries[blackname] = Glicko2Entry(float(row["BlackRating"]), float(row["BlackDeviation"]), float(row["BlackVolatility"]))
		entries[whitename] = Glicko2Entry(float(row["WhiteRating"]), float(row["WhiteDeviation"]), float(row["WhiteVolatility"]))

		line += 1
		if line >= samples:
			break

mse /= line
print(f"mse={mse}")
