# Generate the raw data for the estimate of uncertainty of the strength model.
from strengthnet import StrengthNet
from moves_dataset import MovesDataset

def sample(model, x, y):
  window_size = len(x)
  ys = [0 for i in range(window_size)]  # model outputs for 1..window_size move samples
  for i in range(1, window_size):
    x_est = x[-i:]
    ys[i-1] = model(x_est).to("cpu").item() - y
  return ys

def main(dataset, model, window_size, samplesfile=None):
  count = 0
  accum = [0 for i in range(window_size)]

  model = model.to("cuda")

  if samplesfile:
    samplesfile.write("Moves,Difference\n")

  def examine(x, y):
    nonlocal count
    nonlocal accum
    if window_size > len(x):
      return
    x = x[:window_size].to("cuda")
    ys = sample(model, x, y)
    del x
    if samplesfile:
      for (i, y) in enumerate(ys):
        samplesfile.write(f"{i+1},{y}\n")
    for (i, y) in enumerate(ys):
      accum[i] += y*y
    count += 1

  for (bx, wx, by, wy, _) in dataset:
    examine(bx, by)
    examine(wx, wy)

  for (i, sqsum) in enumerate(accum):
    var = sqsum / count
    sd = math.sqrt(var)
    print(f"{i+1} moves: variance {var:.4f}, sd {sd:.4f}")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Generate the raw data for the estimate of uncertainty of the strength model.")
  parser.add_argument("listpath", type=str, help="Path to the CSV file listing the SGF files and players.")
  parser.add_argument("featurepath", type=str, help="Path to the feature cache directory.")
  parser.add_argument("modelpath", type=str, help="Path to the strength model")
  parser.add_argument("--window-size", type=int, default=500, help="Number of recent moves for the most exact estimate")
  parser.add_argument("--setmarker", choices=["T", "E", "V", "X"], default="T", help="Selected subset of data")
  parser.add_argument("--featurename", choices=["trunk", "pick", "head"], default="pick", help="Feature type to use")
  parser.add_argument("--samplespath", type=str, required=False, help="File to dump raw samples")
  args = parser.parse_args()

  print(vars(args))
  dataset = MovesDataset(args.listpath, args.featurepath, args.setmarker, featurename=args.featurename, featurememory=False)
  model = StrengthNet.load(args.modelpath)
  samplesfile = open(args.samplespath, "w") if args.samplespath else None
  main(dataset, model, args.window_size, samplesfile)
  close(samplesfile)
