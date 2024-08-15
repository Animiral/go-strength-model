#!/usr/bin/env python
# Search the dataset SGF files for board positions that match trickplay patterns.

import os
import csv
import argparse
from sgfmill import sgf, boards
from typing import List, Tuple

def vflip(board: str):
  return "".join([board[19*(18-y) + x] for y in range(19) for x in range(19)])

def hflip(board: str):
  return "".join([board[19*y + 18-x] for y in range(19) for x in range(19)])

def dflip(board: str):
  return "".join([board[19*x + y] for y in range(19) for x in range(19)])

def invcolors(board: str):
  return "".join("x" if "o" == board[i] else "o" if "x" == board[i] else board[i] for i in range(19*19))

def unflip(board: str, symid: str):
  if "i" in symid:
    board = invcolors(board)
  if "d" in symid:
    board = dflip(board)
  if "h" in symid:
    board = hflip(board)
  if "v" in symid:
    board = vflip(board)
  return board

def symmetries(board: str):
  boards = [("", board), ("v", vflip(board))]
  boards = boards + [(s+"h", hflip(b)) for s, b in boards]
  boards = boards + [(s+"d", dflip(b)) for s, b in boards]
  boards = boards + [(s+"i", invcolors(b)) for s, b in boards]
  # eliminate dupes
  return boards

def printboard(board: str):
  for y in range(19):
    print(board[19*y:19*(y+1)])

def readpatterns(patternfiles: List[str]):
  def readpattern(patternfile):
    with open(patternfile, "r") as file:
      boardstr = file.read()
      return boardstr.replace(" ", "").replace("\r", "").replace("\n", "")

  patterns = [(path, symid, boardstr) for path in patternfiles for symid, boardstr in symmetries(readpattern(path))]
  return patterns

def saveboard(sgfpath: str, movenumber: int, symid: str, outpath: str):
  with open(sgfpath, "r") as f:
    sgf_data = f.read()
  game = sgf.Sgf_game.from_string(sgf_data)
  board = boards.Board(19)
  mainseq = game.get_main_sequence()
  for i in range(movenumber+1):
    color, point = mainseq[i].get_move()
    if point is not None:
      row, col = point
      board.play(row, col, color)

  def boardchar(color):
    if color is None:
      return "."
    elif "b" == color:
      return "x"
    elif "w" == color:
      return "o"
    else:
      return "?"

  boardstr = "".join([boardchar(board.get(x, y)) for y in range(19) for x in range(19)])
  boardstr = unflip(boardstr, symid)
  with open(outpath, "w") as f:
    for y in range(19):
      # a bit of pretty-printing like genboard does it
      boardline = " ".join(list(boardstr[19*y:19*(y+1)])).upper() + "\n"
      f.write(boardline)

def checksgf(sgfpath: str, patterns: List[Tuple[str, str]], stonecounts: List[int]):
  """
  Check if any patterns occur in the SGF. This check is incompatible with capture moves.
  As a workaround, the captured location can be marked "?" in the pattern.
  """
  assert len(patterns) == len(stonecounts)
  epatterns = list(enumerate(patterns))

  with open(sgfpath, "r") as f:
    sgf_data = f.read()
  game = sgf.Sgf_game.from_string(sgf_data)
  mainseq = game.get_main_sequence()
  found = []  # tuples of file, movenumber
  mismatches = set()  # collect indexes of patterns invalidated by some stone

  for i, node in enumerate(mainseq):
    color, point = node.get_move()
    if point is None:
      continue

    x, y = point
    index = y*19 + x
    for j, p in epatterns:
      if j in mismatches:
        continue  # disqualified
      path, symid, boardstr = p
      boardchar = boardstr[index]
      if "?" == boardchar:
        continue  # stone doesn't matter here
      elif ("b" == color and "x" == boardchar) or ("w" == color and "o" == boardchar):
        # matches pattern
        newcount = stonecounts[j] - 1
        if 0 == newcount:
          found.append((path, i, symid))
        stonecounts[j] = newcount
      else:
        # violates pattern
        mismatches.add(j)

  return found

def main(listfile: str, patternfiles: List[str]):
  print("Reading File column from " + listfile)

  patterns = readpatterns(patternfiles)  # tuple (filename, patternstr) with dupes for symmetries
  stonecounts = [p.count("x") + p.count("o") for _, _, p in patterns]
  # print([p[0] for p in patterns])

  with open(listfile, "r") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
      sgfpath = row["File"]
      found = checksgf(sgfpath, patterns, stonecounts.copy())
      for patternpath, movenumber, symid in found:
        outpath = patternpath.replace("board", "completed")
        saveboard(sgfpath, movenumber, symid, outpath)
        print(f"{patternpath} in {sgfpath} move {movenumber} sym {symid} -> {outpath}")
      if found:
        patterns = [p for p in patterns if p[0] not in [f[0] for f in found]]  # remove found patterns
        stonecounts = [p.count("x") + p.count("o") for _, _, p in patterns]
      if not patterns:
        break  # early exit once all have been found

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="""
    Search the dataset SGF files for board positions that match trickplay patterns.
    All patterns should contain 'board' in their file name as a convention.
  """)
  parser.add_argument("listfile", type=str, help="Path to the CSV file listing the SGF files.")
  parser.add_argument("patternfiles", type=str, nargs="+", help="Path to the board string files that specify the patterns to search for.")
  args = parser.parse_args()

  main(args.listfile, args.patternfiles)
