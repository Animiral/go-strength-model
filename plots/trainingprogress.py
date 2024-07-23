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
        losses = [[float(token) for token in line.split(',')] for line in file]
    return losses

def plot(trainlosses, testlosses):
    epochs = len(testlosses)

    lt_s = [row[0] for row in trainlosses]
    lt_r = [row[1] for row in trainlosses if len(row) > 1]
    lt_l2 = [row[2] for row in trainlosses if len(row) > 2]
    lt = [sum(l) for l in zip(lt_s, lt_r, lt_l2)]

    lv_s = [row[0] for row in testlosses]
    lv_r = [row[1] for row in testlosses if len(row) > 1]

    trainstep = (epochs-1)/len(trainlosses)
    trainlabels = np.arange(0, epochs-1, trainstep) + trainstep
    testlabels = range(epochs)

    plt.figure(figsize=(12.8, 9.6))

    plt.fill_between(trainlabels, lt_s, color='blue', alpha=0.3, label='Training Score Loss')
    if lt_r:
        plt.fill_between(trainlabels, [sum(x) for x in zip(lt_s, lt_r)], lt_s,
                         color='blue', alpha=0.5, label='Training Ratings Loss')
    if lt_l2:
        plt.fill_between(trainlabels, lt, [sum(x) for x in zip(lt_s, lt_r)],
                         color='blue', alpha=0.7, label='Training Regularization Loss')

    plt.plot(testlabels, lv_s, color='red', linewidth=2, label='Validation Score Loss')
    if lv_r:
        plt.plot(testlabels, lv_r, color='orange', linewidth=2, label='Validation Ratings Loss')

    # plt.plot(trainlabels, trainlosses, label="Training Loss", zorder=1)
    # plt.plot(testlabels, testlosses, label="Validation Loss", zorder=2)  # Ensure test loss is in the foreground
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    trainlossfile = sys.argv[1]
    testlossfile = sys.argv[2]
    print(f"Read train loss from {testlossfile}, validation loss from {testlossfile}.")
    trainlosses = read_losses(trainlossfile)
    testlosses = read_losses(testlossfile)
    plot(trainlosses, testlosses)
