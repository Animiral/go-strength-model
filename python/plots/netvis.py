#!/usr/bin/env python3
# Usage: plots/netvis.py csv/games_labels.csv featurecache --net nets/model.pth --index 10 --featurename pick
"""
Visualize the neural network model layers: activations and gradients.
Visualize the neural network model outputs over the training set vs labels.
"""

import argparse
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from model.moves_dataset import MovesDataset, MovesDataLoader
from model.strengthnet import StrengthNet
import fontconfig

device = "cpu"

def main(listpath, featurepath, featurename, netpath, index):
  data = MovesDataset(listpath, featurepath, "T", featurename=featurename)

  if netpath:
    model = StrengthNet.load(netpath)
  else:
    model = newmodel(data.featureDims)
  model = model.to(device)
  print(f"Parameters: {sum(p.nelement() for p in model.parameters())}") # number of parameters in total

  bx, wx, by, wy, s = data[index]  # blackRecent, whiteRecent, game.black.rating, game.white.rating, game.score
  print(f"Game {index}: {len(bx)} black recent, {len(wx)} white recent, by={by}, wy={wy}, s={s}")
  bx, by = bx.to(device), torch.tensor(by).to(device)
  bpred = model(bx)
  model.retain_grads()
  MSE = nn.MSELoss()
  loss = MSE(bpred, by)
  loss.backward()

  fig, axs = plt.subplots(2, 3, figsize=(12.8, 9.6))

  emb_lines = setup_embeddings(axs[0, 0], model)
  act_alines, act_hlines = setup_activations(axs[0, 1], axs[0, 2], model)
  grad_alines, grad_hlines = setup_gradients(axs[1, 0], axs[1, 1], model)
  plot_embeddings(axs[0, 0], emb_lines, model)
  plot_activations(axs[0, 1], axs[0, 2], act_alines, act_hlines, model)
  plot_gradients(axs[1, 0], axs[1, 1], grad_alines, grad_hlines, model)
  plot_outputs(data, model)
  plt.show()

def setup_embeddings(ax, model):
  ax.set_ylabel("Density")
  ax.set_title("Layer Embeddings")
  lines = []

  for l in range(model.depth + 2):  # ISABs, input and pooling layers
    # print(f"Embeddings {l} mean: {h.mean()} stdev: {h.std()}")
    histline, = ax.plot([], [], alpha=0.7, label=f"Layer {l}")
    meanline = ax.axvline(0, color=histline.get_color(), linestyle='--', linewidth=2)
    stdline1 = ax.axvline(0, color=histline.get_color(), linestyle='-.', linewidth=1)
    stdline2 = ax.axvline(0, color=histline.get_color(), linestyle='-.', linewidth=1)
    lines.append((histline, meanline, stdline1, stdline2))

  ax.legend()
  return lines

def setup_activations(a_ax, h_ax, model):
  a_ax.set_ylim(0, 50)
  a_ax.set_ylabel("Density")
  a_ax.set_title("Softmax Activations")
  alines = []

  for l in range(model.depth):  # ISABs
    line, = a_ax.plot([], [], label=f"Layer {l+1}")
    alines.append(line)

  a_ax.legend()

  h_ax.set_ylabel("Density")
  h_ax.set_title("ReLu Preactivations")
  hlines = []

  for l in range(model.depth):  # ISABs
    line, = h_ax.plot([], [], label=f"Layer {l+1}")
    hlines.append(line)

  h_ax.legend()

  return alines, hlines

def setup_gradients(a_ax, h_ax, model):
  a_ax.set_ylabel("Density")
  a_ax.set_title("Softmax Grads")
  alines = []

  for l in range(model.depth):  # ISABs
    line, = a_ax.plot([], [], label=f"Layer {l+1}")
    alines.append(line)

  a_ax.legend()

  h_ax.set_ylabel("Density")
  h_ax.set_title("ReLu Grads")
  hlines = []

  for l in range(model.depth):  # ISABs
    line, = h_ax.plot([], [], label=f"Layer {l+1}")
    hlines.append(line)

  h_ax.legend()

  return alines, hlines

def plot_embeddings(ax, lines, model):
  hs = model.embeddings()

  for l, h in enumerate(hs):
    # print(f"Embeddings {l} mean: {h.mean()} stdev: {h.std()}")
    hy, hx = torch.histogram(h.cpu(), density=True)
    histline, meanline, stdline1, stdline2 = lines[l]
    histline.set_xdata(hx[:-1].detach())
    histline.set_ydata(hy.detach())
    mean = h.mean().item()
    std = h.std().item()
    meanline.set_xdata([mean, mean])
    stdline1.set_xdata([mean - std, mean - std])
    stdline2.set_xdata([mean + std, mean + std])

  ax.relim()
  ax.autoscale_view()

def plot_activations(a_ax, h_ax, alines, hlines, model):
  a_acts, h_acts = model.activations()

  for l, act in enumerate(a_acts):
    hy, hx = torch.histogram(act.cpu(), density=True)
    line = alines[l]
    line.set_xdata(hx[:-1].detach())
    line.set_ydata(hy.detach())

  for l, act in enumerate(h_acts):
    hy, hx = torch.histogram(act.cpu(), density=True)
    line = hlines[l]
    line.set_xdata(hx[:-1].detach())
    line.set_ydata(hy.detach())

  for ax in a_ax, h_ax:
    ax.relim()
    ax.autoscale_view()

def plot_gradients(a_ax, h_ax, alines, hlines, model):
  grads = model.grads()
  if grads is None:
    return  # do not plot if model has no backprop data

  a_grads, h_grads = grads

  for l, grad in enumerate(a_grads):
    hy, hx = torch.histogram(grad.cpu(), density=True)
    line = alines[l]
    line.set_xdata(hx[:-1].detach())
    line.set_ydata(hy.detach())

  for l, grad in enumerate(h_grads):
    hy, hx = torch.histogram(grad.cpu(), density=True)
    line = hlines[l]
    line.set_xdata(hx[:-1].detach())
    line.set_ydata(hy.detach())

  for ax in a_ax, h_ax:
    ax.relim()
    ax.autoscale_view()

def plot_outputs(data, model):
  outs = []
  labels = []
  progress = 0
  print(f"Evaluate training set of size {len(data)}")

  model.eval()
  for (bx, wx, by, wy, s) in data:  # blackRecent, whiteRecent, game.black.rating, game.white.rating, game.score
    print(".", end="", flush=True)
    labels.append(by)
    labels.append(wy)
    if progress < 20:  # temp limit
      with torch.no_grad():
        outs.append(model(bx.to(device)).item())
        outs.append(model(wx.to(device)).item())
    progress += 1

  print()
  print(f"Outputs mean: {np.mean(outs)} stdev: {np.std(outs)}")
  print(f"Labels mean: {np.mean(labels)} stdev: {np.std(labels)}")

  hy, hx = np.histogram(outs, bins="auto", density=True)
  plt.plot(hx[:-1], hy, label=f"Model Output")
  hy, hx = np.histogram(labels, bins="auto", density=True)
  plt.plot(hx[:-1], hy, label=f"Training Labels")

  plt.ylabel("Density")
  plt.title("Output Distribution")
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
