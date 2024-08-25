#!/usr/bin/env python3
# Repair bugged training loss records with duplicate rows

def readfile(path):
  with open(path, "r") as file:
    rows = [tuple(float(token) for token in line.split(',')) for line in file]
  return rows

def countepochs(rows, maxepochs, steps):
  """Find out how many epochs actually fit in rows, assuming the rowcount is bugged"""
  for e in range(1, maxepochs):
    brokencount = (e * (e+1)) / 2 * steps
    if len(rows) == brokencount:
      return e
    if len(rows) < brokencount:  # could not find the nr of epochs
      return None
  return None

def checkdupes(rows, epochs, steps):
  """
  Check if the rows actually match the dupe bug that we want to fix.
  Epoch 1 data must match beginning of real data.
  Second-to-last epoch data must match most of real data.
  """
  realrows = rows[-epochs*steps:]
  for a, b in zip(rows[:steps], realrows):
    assert a == b
  prelastdata = rows[-2*epochs*steps + steps:-epochs*steps]
  for a, b in zip(prelastdata, realrows):
    assert a == b
  return realrows

def writefile(path, rows):
  with open(path, "w") as file:
    for row in rows:
      file.write(",".join(str(v) for v in row) + "\n")

def fixfile(path, maxepochs, steps):
  print(f"Fixing {path}... ", end="")
  rows = readfile(path)
  epochs = countepochs(rows, maxepochs, steps)
  # import pdb; pdb.set_trace()
  if epochs is None:
    print(f"Could not determine number of epochs for {len(rows)} rows. File might not be bugged.")
    return
  realrows = checkdupes(rows, epochs, steps)
  writefile(path, realrows)
  print("Done.")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Repair bugged training loss records with duplicate rows.")
  parser.add_argument("--epochs", type=int, default=100, help="Expected maximum number of epochs")
  parser.add_argument("--steps", type=int, default=100, help="Expected number of steps per epoch")
  parser.add_argument("paths", type=str, nargs="+", help="Loss record files to be fixed")
  args = parser.parse_args()

  for path in args.paths:
    epochs = args.epochs
    steps = args.steps
    try:
      fixfile(path, epochs, steps)
    except Exception as ex:
      print(ex)
