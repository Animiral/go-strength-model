import argparse
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler
from moves_dataset import MovesDataset, MovesDataLoader
from strengthnet import StrengthNet
from torch.nn.utils.rnn import pack_padded_sequence

device = "cuda"

def main(args):
    listfile = args["listfile"]
    featuredir = args["featuredir"]
    featurename = args["featurename"]
    outfile = args["outfile"]
    trainlossfile = args["trainlossfile"]
    testlossfile = args["testlossfile"]
    batch_size = args["batch_size"]
    steps = args["steps"]
    epochs = args["epochs"]

    print(f"Load training data from {listfile}")
    print(f"Load precomputed {featurename} features from {featuredir}")
    print(f"Save model(s) to {outfile}")
    print(f"Batch size: {batch_size}")
    print(f"Steps: {steps}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")

    if trainlossfile:
        print(f"Write training loss to {trainlossfile}")
        trainlossfile = open(trainlossfile, 'w')
    if testlossfile:
        print(f"Write validation loss to {testlossfile}")
        testlossfile = open(testlossfile, 'w')

    train_data = MovesDataset(listfile, featuredir, 'T', featurename=featurename)
    test_data = MovesDataset(listfile, featuredir, 'V', featurename=featurename)
    test_loader = MovesDataLoader(test_data, batch_size=batch_size)

    model = StrengthNet(train_data.featureDims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if outfile:
        outpath = outfile.replace('{}', "0")
        torch.save(model.state_dict(), outpath)

    for e in range(epochs):
        sampler = BatchSampler(RandomSampler(train_data, replacement=True, num_samples=steps*batch_size), batch_size, False)
        train_loader = MovesDataLoader(train_data, batch_sampler=sampler)
        print(f"Epoch {e+1}\n-------------------------------")

        trainloss = train(train_loader, model, optimizer, steps*batch_size)
        if trainlossfile:
            for loss in trainloss:
                trainlossfile.write(f"{loss}\n")

        testloss = test(test_loader, model)
        if testlossfile:
            testlossfile.write(f"{testloss}\n")

        if outfile:
            outpath = outfile.replace('{}', str(e+1))
            torch.save(model.state_dict(), outpath)

    trainlossfile and trainlossfile.close()
    testlossfile and testlossfile.close()
    print("Done!")

def train(loader, model, optimizer, totalsize: int=0):
    samples = 0  # how many we have learned
    model.train()
    MSE = nn.MSELoss()
    trainloss = []

    for batchnr, (bx, wx, blens, wlens, by, wy, score) in enumerate(loader):
        bx, by, wx, wy = map(lambda t: t.to(device), (bx, by, wx, wy))
        bpred, wpred = model(bx, blens), model(wx, wlens)
        loss = MSE(bpred, by) + MSE(wpred, wy) # + crossentropy(bt(bpred, wpred), score)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # status
        batch_size = len(score)
        loss = loss.item() / batch_size
        trainloss.append(loss)
        samples += batch_size
        print(f"loss: {loss:>7f}  [{samples:>5d}/{totalsize:>5d}]")

    return trainloss

def test(loader, model):
    batches = len(loader)
    model.eval()
    MSE = nn.MSELoss()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for bx, wx, blens, wlens, by, wy, score in loader:
            bx, by, wx, wy = map(lambda t: t.to(device), (bx, by, wx, wy))
            bpred, wpred = model(bx, blens), model(wx, wlens)
            loss = MSE(bpred, by) + MSE(wpred, wy)
            test_loss += loss.item()
    test_loss /= batches
    print(f"Validation Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

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
    required_args.add_argument('featuredir', help='Directory containing extracted features')
    optional_args.add_argument('-f', '--featurename', help='Type of features to train on', type=str, default='pick', required=False)
    optional_args.add_argument('-o', '--outfile', help='Pattern for model output, with epoch placeholder "{}" ', type=str, required=False)
    optional_args.add_argument('-b', '--batch-size', help='Minibatch size', type=int, default=100, required=False)
    optional_args.add_argument('-t', '--steps', help='Number of batches per epoch', type=int, default=100, required=False)
    optional_args.add_argument('-e', '--epochs', help='Nr of training epochs', type=int, default=5, required=False)
    optional_args.add_argument('--trainlossfile', help='Output file to store training loss values', type=str, required=False)
    optional_args.add_argument('--testlossfile', help='Output file to store validation loss values', type=str, required=False)

    args = vars(parser.parse_args())
    main(args)
