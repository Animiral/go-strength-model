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

def setup(ax, epochs=0):
    if epochs:
        ax.set_xlim(0, epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")

def plot(ax, trainlosses, testlosses):
    """Draw the plot data in the given axis"""
    if len(trainlosses) > 1:
        lt_s = [row[0] for row in trainlosses]
        lt_r = [row[1] for row in trainlosses if len(row) > 1]
        lt_l2 = [row[2] for row in trainlosses if len(row) > 2]
        lt = [sum(l) for l in zip(lt_s, lt_r, lt_l2)]

    lv_s = [row[0] for row in testlosses]
    lv_r = [row[1] for row in testlosses if len(row) > 1]

    if len(trainlosses) > 1:
        trainlabels = (np.arange(len(trainlosses)) + 1) * (len(testlosses) - 1) / len(trainlosses)
    testlabels = range(len(testlosses))

    if len(trainlosses) > 1:
        ax.fill_between(trainlabels, lt_s, color='blue', alpha=0.3, label="Training Score Loss")
        ax.fill_between(trainlabels, [sum(x) for x in zip(lt_s, lt_r)], lt_s, color='blue', alpha=0.5, label="Training Ratings Loss")
        ax.fill_between(trainlabels, lt, [sum(x) for x in zip(lt_s, lt_r)], color='blue', alpha=0.7, label="Training Regularization Loss")
    ax.plot(testlabels, lv_s, color='red', linewidth=2, label="Validation Score Loss")
    ax.plot(testlabels, lv_r, color='orange', linewidth=2, label="Validation Ratings Loss")
    ax.legend()

if __name__ == "__main__":
    trainlossfile = sys.argv[1]
    testlossfile = sys.argv[2]
    print(f"Read train loss from {testlossfile}, validation loss from {testlossfile}.")
    trainlosses = read_losses(trainlossfile)
    testlosses = read_losses(testlossfile)

    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111)
    setup(ax)
    plot(ax, trainlosses, testlosses)
    plt.show()
