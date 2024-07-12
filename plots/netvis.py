#!/usr/bin/env python3
# Usage: plots/netvis.py csv/games_labels.csv featurecache --net nets/model.pth --index 10 --featurename pick
"""
Visualize the neural network model layers: activations and gradients.
"""

import argparse
import torch
import matplotlib.pyplot as plt
from model.moves_dataset import MovesDataset, MovesDataLoader
from model.strengthnet import StrengthNet

device = "cpu"

def main(listpath, featurepath, featurename, netpath, index):
  data = MovesDataset(listpath, featurepath, 'T', featurename=featurename)
  model = newmodel(data.featureDims).to(device)

  if netpath:
    model.load_state_dict(torch.load(netpath)).to(device)

  bx, _, _, _, _ = data[index]
  bx = bx.to(device)
  bpred = model(bx).item()
  a_acts, h_acts = model.activations()

  for l, act in enumerate(a_acts):
    hy, hx = torch.histogram(act, density=True)
    plt.plot(hx[:-1].detach(), hy.detach(), label=f"Layer {l}")

  plt.ylim(0, 50)
  plt.ylabel('Density')
  plt.title('Softmax Activations')
  plt.legend()
  plt.show()

  for l, act in enumerate(h_acts):
    hy, hx = torch.histogram(act, density=True)
    plt.plot(hx[:-1].detach(), hy.detach(), label=f"Layer {l}")

  # plt.xticks([])
  plt.ylabel('Density')
  plt.title('ReLu Preactivations')
  plt.legend()
  plt.show()

def newmodel(featureDims: int):
    depth = 5
    hiddenDims = 16
    queryDims = 8
    inducingPoints = 8
    return StrengthNet(featureDims, depth, hiddenDims, queryDims, inducingPoints)

if __name__ == "__main__":
  description = "Visualize the neural network model layers: activations and gradients."
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("listpath", type=str, help="Path to the dataset games list")
  parser.add_argument("featurepath", type=str, help="Path to the precomputed feature cache")
  parser.add_argument("-f", "--featurename", help="Type of features to train on", type=str, default="pick", required=False)
  parser.add_argument("--net", default="", type=str, help="Path to the strength network weights file", required=False)
  parser.add_argument("--index", default=0, type=int, help="Index into the training set indicating which sample to feed into the net", required=False)
  args = vars(parser.parse_args())
  print(args)
  main(args["listpath"], args["featurepath"], args["featurename"], args["net"], int(args["index"]))
