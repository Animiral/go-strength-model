import sys
import numpy as np
import matplotlib.pyplot as plt
import fontconfig

def read_losses(filename):
    with open(filename, "r") as file:
        losses = [float(line.strip()) for line in file]
    return losses

def plot(trainlosses, testlosses):
    plt.figure(figsize=(12.8, 9.6))
    epochs = len(testlosses)
    trainstep = (epochs-1)/len(trainlosses)
    trainlabels = np.arange(0, epochs-1, trainstep) + trainstep
    testlabels = range(epochs)
    plt.plot(trainlabels, trainlosses, label="Training Loss", zorder=1)
    plt.plot(testlabels, testlosses, label="Validation Loss", zorder=2)  # Ensure test loss is in the foreground
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
