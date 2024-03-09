import sys
import matplotlib.pyplot as plt

def read_losses(filename):
    with open(filename, 'r') as file:
        losses = [float(line.strip()) for line in file]
    return losses

def aggregate_losses(losses, num_epochs):
    interval = len(losses) / num_epochs
    aggregated_losses = []
    for i in range(num_epochs):
        start = int(i * interval)
        end = int((i + 1) * interval)
        aggregated_losses.append(sum(losses[start:end]) / (end - start))
    return aggregated_losses

def plot(train_losses, test_losses):
    plt.figure(figsize=(5, 3))
    plt.plot(train_losses, label='Training Loss', zorder=1)
    plt.plot(test_losses, label='Validation Loss', zorder=2)  # Ensure test loss is in the foreground
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    trainlossfile = sys.argv[1]
    testlossfile = sys.argv[2]
    print(f'Read train loss from {testlossfile}, validation loss from {testlossfile}.')
    train_losses = read_losses(trainlossfile)
    test_losses = read_losses(testlossfile)
    num_epochs = len(test_losses)
    aggregated_train_losses = aggregate_losses(train_losses, num_epochs)
    plot(aggregated_train_losses, test_losses)
