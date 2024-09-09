# Examine logs of hyperparameter search for statistics, specifically early stop time.
import sys
import os

def examine(logfile):
  epochs = 0
  with open(logfile, "r") as file:
    for line in file:
      if "Epoch" in line:
        epochs += 1
  print(f"{os.path.basename(logfile)}: {epochs-1} epochs")

if __name__ == "__main__":
  """
  Given a path to the finished hyperparameter search log directory,
  examine the logs for statistics, specifically early stop time.
  Output to stdout.
  """
  searchdir = sys.argv[1]
  print(f"Read logs from {searchdir}.")

  for logfile in os.listdir(searchdir):
    if logfile.startswith("training_"):
      examine(os.path.join(searchdir, logfile))
