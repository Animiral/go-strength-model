#!/usr/bin/env python3
# Usage: plots/trainingprogress.py logs/trainloss.txt logs/validationloss.txt
"""
Reads a `trainloss` and a `validationloss` file and draws them.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import fontconfig

def read_losses(filename):
    with open(filename, "r") as file:
        losses = [tuple(float(token) for token in line.split(',')) for line in file]
    return losses

def setup(ax, epochs=0, title=None):
    if epochs:
        ax.set_xlim(0, epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    pretitle = f"{title}: " if title is not None else ""
    ax.set_title(f"{pretitle}Training and Validation Loss")

def plot(ax, trainlosses, testlosses):
    """Draw the plot data in the given axis"""
    if len(trainlosses) > 1:
        lt_s = [row[0] for row in trainlosses]
        lt_r = [row[1] for row in trainlosses if len(row) > 1]
        lt_l2 = [row[2] for row in trainlosses if len(row) > 2]
        lt_mid = [sum(x) for x in zip(lt_s, lt_r)]
        lt = [sum(l) for l in zip(lt_s, lt_r, lt_l2)]

    lv_s = [row[0] for row in testlosses]
    lv_r = [row[1] for row in testlosses if len(row) > 1]

    if len(trainlosses) > 1:
        trainlabels = (np.arange(len(trainlosses)) + 1) * (len(testlosses) - 1) / len(trainlosses)
    testlabels = range(len(testlosses))

    if len(trainlosses) > 1:
        ax.fill_between(trainlabels, lt_s, color='lightblue', label="Training Score Loss")
        ax.fill_between(trainlabels, lt_mid, lt_s, color='#0A9EF8', label="Training Ratings Loss")
        ax.fill_between(trainlabels, lt, lt_mid, color='darkblue', label="Training Regularization Loss")
    ax.plot(testlabels, lv_s, color='red', linewidth=2, label="Validation Score Loss")
    ax.plot(testlabels, lv_r, color='orange', linewidth=2, label="Validation Ratings Loss")
    ax.set_ylim(0, 3.8)
    ax.legend()

if __name__ == "__main__":
    trainlossfile = sys.argv[1]
    testlossfile = sys.argv[2]
    if 4 == len(sys.argv):
        title = sys.argv[3]
    else:
        title = None

    if title:
        print(f"Title: {title}")
    print(f"Read train loss from {trainlossfile}, validation loss from {testlossfile}.")

    trainlosses = read_losses(trainlossfile)
    testlosses = read_losses(testlossfile)

    fig = plt.figure(figsize=fontconfig.ideal_figsize)
    ax = fig.add_subplot(111)
    setup(ax, title=title)
    plot(ax, trainlosses, testlosses)
    plt.show()
