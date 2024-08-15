#!/usr/bin/env python3
"""
Usage: ogsfit.py INPUTFILE

Given a CSV file containing ratings and predictions for games, fit the
predictions x to the ratings y using the model f(x|a,b) = ax + b,
minimizing MSE(y, f(x)).
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def main(inputpath):
  data = pd.read_csv(inputpath)
  x = np.concatenate([data["PredictedBlackRating"].values, data["PredictedWhiteRating"].values])
  y = np.concatenate([data["BlackRating"].values, data["WhiteRating"].values])

  def mse(params, x, y):
      a, b = params
      y_hat = a * x + b
      return np.mean((y - y_hat) ** 2)

  initial_guess = [315.8087941393861, 1623.1912875700598]  # a, b
  result = minimize(mse, initial_guess, args=(x, y))
  a, b = result.x
  print(f"Best scaling parameters: rating = {a} * model_y + {b}")

if __name__ == "__main__":
  """
  Given a CSV file containing ratings and predictions for games, fit the
  predictions x to the ratings y using the model f(x|a,b) = ax + b,
  minimizing MSE(y, f(x)).
  """
  inputpath = sys.argv[1]
  print(f"Read OGS ratings and model ratings from {inputpath}.")
  main(inputpath)
