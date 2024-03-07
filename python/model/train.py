import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

device = "cuda"
training_data = [torch.rand((5, 10, 3))]
training_labels = [torch.rand(5)]
test_data = [torch.rand((5, 10, 3))]
test_labels = [torch.rand(5)]

class StrengthNet(nn.Module):
    def __init__(self):
        d_hidden = 32
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(3, d_hidden),
            nn.ReLU()
        )
        self.rating = nn.Linear(d_hidden, 1)
        self.weights = nn.Linear(d_hidden, 1)

    def forward(self, x):
        h = self.layer1(x)
        r = self.rating(h).squeeze(-1)
        z = self.weights(h).squeeze(-1)
        w = F.softmax(z, dim=1)
        return torch.sum(r * w, dim=1)

def main(args):
    listfile = args["listfile"]
    batch_size = args["batch_size"]
    epochs = args["epochs"]

    print(f"Load training data from {listfile}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")

    model = StrengthNet().to(device)
    print(f"Model parameters: {list(model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(list(zip(training_data, training_labels)), model, optimizer)
        test(list(zip(test_data, test_labels)), model)
    print("Done!")

def train(data, model, optimizer):
    size = len(data)
    model.train()
    MSE = nn.MSELoss()

    for batch, (X, y) in enumerate(data):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = MSE(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # status
        print(f"loss: {loss.item():>7f}  [{batch:>5d}/{size:>5d}]")

def test(data, model):
    size = len(data)
    model.eval()
    MSE = nn.MSELoss()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += MSE(pred, y).item()
    test_loss /= size
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    description = """
    Train strength model on Go positions from dataset.
    """

    parser = argparse.ArgumentParser(description=description,add_help=False)
    required_args = parser.add_argument_group('required arguments')
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument(
        '-h',
        '--help',
        action='help',
        default=argparse.SUPPRESS,
        help='show this help message and exit'
    )
    required_args.add_argument('listfile', help='CSV file listing games and labels')
    optional_args.add_argument('-b', '--batch-size', help='Minibatch size', type=int, default=100, required=False)
    optional_args.add_argument('-t', '--epochs', help='Nr of training epochs', type=int, default=5, required=False)

    args = vars(parser.parse_args())
    main(args)
