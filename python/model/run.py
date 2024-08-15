#!/usr/bin/env python3
# Apply the strength model on one or multiple SGF files, including KataGo preprocessing.

import argparse
import subprocess
import sys
import tempfile
import torch
from moves_dataset import load_features_from_zip, scale_rating
from strengthnet import StrengthNet
from rank import *

device = "cuda"

class KatagoException(Exception):
    pass

def main(args):
    inputs = args["inputs"]
    katapath = args["katago"]
    katamodel = args["katamodel"]
    kataconfig = args["kataconfig"]
    modelfile = args["model"]
    featurename = args["featurename"]
    playername = args["playername"]
    scale = args["scale"]

    sgfs = [p for p in inputs if p.endswith(".sgf")]
    zips = [p for p in inputs if not p in sgfs]
    if sgfs:
        print(f"Evaluating {len(sgfs)} game records:")
        for path in sgfs:
            print(f"  - {path}")
    if zips:
        print(f"Evaluating {len(zips)} feature archives:")
        for path in zips:
            print(f"  - {path}")
    print(f"KataGo binary: {katapath}")
    print(f"KataGo model: {katamodel}")
    print(f"KataGo configuration: {kataconfig if kataconfig else '(default)'}")
    print(f"Strength model: {modelfile}")
    print(f"Extract \"{featurename}\" features.")
    print(f"Player name: {playername if playername else '(auto-detect)'}")
    print(f"Scale: {str(scale) if scale else '(default)'}")
    print(f"Device: {device}")

    if sgfs:
        with tempfile.NamedTemporaryFile(suffix='.zip') as outFile:
            print(f"Feature file: {outFile.name}")
            print(f"Executing katago...")
            xs = katago(katapath, katamodel, kataconfig, sgfs, outFile.name, featurename, playername)

    print(f"Loading strength model...")

    model = StrengthNet.load(modelfile).to(device)

    print(f"Executing strength model...")
    if scale:
        scaling_func = lambda r: scale[0] * r + scale[1]
    else:
        scaling_func = scale_rating

    results = []

    if sgfs:
        assert model.featureDims == xs.shape[1]  # strength model must fit KataGo model
        pred = evaluate(xs, model)
        pred = scaling_func(pred)
        rank = to_rank(pred)
        print(f"SGFs (player: {playername}): {pred} ({rankstr(rank)})")
        results.append((None, pred, rankstr(rank)))

    for z in zips:
        xs = load_features_from_zip(z, featurename)
        assert model.featureDims == xs.shape[1]  # strength model must fit KataGo model
        pred = evaluate(xs, model)
        pred = scaling_func(pred)
        rank = to_rank(pred)
        print(f"{z}: {pred} ({rankstr(rank)})")
        results.append((z, pred, rankstr(rank)))

    return results

def newmodel(featureDims: int, args):
    depth = args.get("modeldepth", 2)
    hiddenDims = args.get("hidden_dims", 16)
    queryDims = args.get("query_dims", 8)
    inducingPoints = args.get("inducing_points", 8)
    return StrengthNet(featureDims, depth, hiddenDims, queryDims, inducingPoints)

def katago(binPath: str, modelPath: str, configPath: str, sgfFiles: list[str], outFile: str, featureName: str, playerName: str) -> torch.Tensor:
    selection = f"-with-{featureName}"
    playerNameArg = ["-playername", playerName] if playerName else ["-autodetect"]
    command = [binPath, "extract_sgfs"] + sgfFiles + ["-model", modelPath, "-config", configPath, "-outfile", outFile, selection] + playerNameArg

    # temp debug
    print("kata cmd: " + " ".join(command))

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            print(line, end="", file=sys.stdout)
        for line in process.stderr:
            print(line, end="", file=sys.stderr)
        
        process.wait()
        if process.returncode != 0:
            raise KatagoException(f"KataGo execution failed with return code {process.returncode}.")

    return load_features_from_zip(outFile, featureName)

def evaluate(xs: torch.Tensor, model: StrengthNet):
    model.eval()
    with torch.no_grad():
        xs = xs.to(device)
        pred = model(xs).item()
    return pred

if __name__ == "__main__":
    description = """
    Apply the strength model on one or multiple SGF or ZIP files.
    SGFs will be combined towards one output strength estimate.
    ZIPs will be judged each by itself.
    """
    parser = argparse.ArgumentParser(description=description,add_help=False)
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="show this help message and exit")
    parser.add_argument("inputs", type=str, nargs="+", help="SGF or ZIP file(s) to evaluate")
    parser.add_argument("--katamodel", help="KataGo neural network weights file", required=False)
    parser.add_argument("--model", help="Strength model neural network weights file", required=False)
    parser.add_argument("-b", "--katago", help="Path to katago binary", type=str, default="katago", required=False)
    parser.add_argument("-c", "--kataconfig", help="Path to katago configuration", type=str, required=False)
    parser.add_argument("-f", "--featurename", help="Type of features to use", type=str, default="pick", required=False)
    parser.add_argument("-p", "--playername", help="SGF player name to evaluate", type=str, required=False)
    parser.add_argument("-s", "--scale", help="2 coefficients for scaling the model output to Glicko-2", type=float, nargs=2, required=False)

    args = vars(parser.parse_args())
    main(args)
