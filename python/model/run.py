#!/usr/bin/env python3
# Apply the strength model on one or multiple SGF files, including KataGo preprocessing.

import argparse
import subprocess
import sys
import tempfile
import torch
from moves_dataset import load_features_from_zip, scale_rating
from strengthnet import StrengthNet

device = "cuda"

def main(args):
    sgfs = args["sgf"]
    katapath = args["katago"]
    katamodel = args["katamodel"]
    kataconfig = args["kataconfig"]
    modelfile = args["model"]
    featurename = args["featurename"]
    playername = args["playername"]

    print(f"Evaluating {len(sgfs)} game records:")
    for sgf in sgfs:
        print(f"  - {sgf}")
    print(f"KataGo binary: {katapath}")
    print(f"KataGo model: {katamodel}")
    print(f"KataGo configuration: {kataconfig if kataconfig else '(default)'}")
    print(f"Strength model: {modelfile}")
    print(f"Extract \"{featurename}\" features.")
    print(f"Player name: {playername if playername else '(auto-detect)'}")
    print(f"Device: {device}")

    with tempfile.NamedTemporaryFile(suffix='.zip') as outFile:
        print(f"Feature file: {outFile.name}")
        print(f"Executing katago...")
        xs = katago(katapath, katamodel, kataconfig, sgfs, outFile.name, featurename, playername)

    print(f"Loading strength model...")

    model = StrengthNet.load(modelfile).to(device)
    assert(model.featureDims == xs.shape[1])  # strength model must fit KataGo model

    print(f"Executing strength model...")

    pred = evaluate(xs, model)
    pred = scale_rating(pred)

    print(f"The estimated rating of {playername} is {pred}.")

def newmodel(featureDims: int, args):
    depth = args.get("modeldepth", 2)
    hiddenDims = args.get("hidden_dims", 16)
    queryDims = args.get("query_dims", 8)
    inducingPoints = args.get("inducing_points", 8)
    return StrengthNet(featureDims, depth, hiddenDims, queryDims, inducingPoints)

def katago(binPath: str, modelPath: str, configPath: str, sgfFiles: list[str], outFile: str, featureName: str, playerName: str) -> torch.Tensor:
    selection = f"-with-{featureName}"
    playerNameArg = ["-playername", playerName] if playerName else []
    command = [binPath, "extract_sgfs"] + sgfFiles + ["-model", modelPath, "-config", configPath, "-outfile", outFile, selection] + playerNameArg

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end="", file=sys.stdout)
        for line in process.stderr:
            print(line, end="", file=sys.stderr)
        
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"KataGo execution failed with return code {process.returncode}.")

    return load_features_from_zip(outFile, featureName)

def evaluate(xs: torch.Tensor, model: StrengthNet):
    model.eval()
    with torch.no_grad():
        xs = xs.to(device)
        pred = model(xs).item()
    return pred

if __name__ == "__main__":
    description = """
    Apply the strength model on one or multiple SGF files.
    """

    parser = argparse.ArgumentParser(description=description,add_help=False)
    required_args = parser.add_argument_group("required arguments")
    optional_args = parser.add_argument_group("optional arguments")
    optional_args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit"
    )
    required_args.add_argument("sgf", type=str, nargs="+", help="SGF file(s) to evaluate")
    required_args.add_argument("--katamodel", help="KataGo neural network weights file")
    required_args.add_argument("--model", help="Strength model neural network weights file")
    optional_args.add_argument("-b", "--katago", help="Path to katago binary", type=str, default="katago", required=False)
    optional_args.add_argument("-c", "--kataconfig", help="Path to katago configuration", type=str, required=False)
    optional_args.add_argument("-f", "--featurename", help="Type of features to use", type=str, default="pick", required=False)
    optional_args.add_argument("-p", "--playername", help="SGF player name to evaluate", type=str, required=False)

    args = vars(parser.parse_args())
    main(args)
