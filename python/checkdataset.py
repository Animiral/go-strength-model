#!/usr/bin/env python3
# Read a list file (dataset) and print out the marked rows and available features.

import os
from model.moves_dataset import MovesDataset, GameEntry

def countRecentMoves(dataset: MovesDataset, player: str, game: GameEntry, featurename: str):
    try:
        data = dataset.loadRecentMoves(player, game, featurename)
        return len(data)
    except (FileNotFoundError, KeyError):
        return 0

def statRecentMoves(dataset: MovesDataset, player:str, game: GameEntry):
    nhead = countRecentMoves(dataset, player, game, "head")
    npick = countRecentMoves(dataset, player, game, "pick")
    ntrunk = countRecentMoves(dataset, player, game, "trunk")
    count = nhead or npick or ntrunk

    # check count for consistency
    if npick != 0 and npick != count:
        raise ValueError("Inconsistent number of recent moves!")
    if ntrunk != 0 and ntrunk != count:
        raise ValueError("Inconsistent number of recent moves!")

    haveHead = nhead > 0
    havePick = npick > 0
    haveTrunk = ntrunk > 0
    return count, haveHead, havePick, haveTrunk

def main(listpath: str, featuredir: str, require_head: bool = False, require_pick: bool = False, require_trunk: bool = False):
    dataset = MovesDataset(listpath, featuredir, "*", sparse=False)
    print(f"Loaded dataset {listpath} with {len(dataset.games)} games.")

    tcount = 0
    vcount = 0
    ecount = 0
    errors = []  # records without recent move data or inconsistent count
    anyHead = False  # at least one record has head features
    anyPick = False  # at least one record has pick features
    anyTrunk = False  # at least one record has trunk features
    allHead = True  # all records have head features
    allPick = True  # all records have pick features
    allTrunk = True  # all records have trunk features

    for game in dataset.games:
        if game.marker not in {"T", "V", "E"}:
            continue

        if "T" == game.marker:
            tcount += 1
        if "V" == game.marker:
            vcount += 1
        if "E" == game.marker:
            ecount += 1

        bname = game.black.name
        brating = game.black.rating
        wname = game.white.name
        wrating = game.white.rating

        try:
            bcount, bHead, bPick, bTrunk = statRecentMoves(dataset, "Black", game)
            wcount, wHead, wPick, wTrunk = statRecentMoves(dataset, "White", game)
        except ValueError as e:
            print(f"{bname} ({brating}) vs {wname} ({wrating}): {game.score}, {e}")
            errors.append((game, str(e)))  # record with inconsistent count
            continue

        if require_head and not (bHead and wHead):
            errors.append((game, "missing required head features"))
        if require_pick and not (bPick and wPick):
            errors.append((game, "missing required pick features"))
        if require_trunk and not (bTrunk and wTrunk):
            errors.append((game, "missing required trunk features"))

        anyHead = anyHead or bHead or wHead
        anyPick = anyPick or bPick or wPick
        anyTrunk = anyTrunk or bTrunk or wTrunk
        allHead = allHead and bHead and wHead
        allPick = allPick and bPick and wPick
        allTrunk = allTrunk and bTrunk and wTrunk
        if 0 == bcount and 0 == wcount:
            errors.append((game, "no recent moves"))  # record without recent move data
        bfeatures = ("h" if bHead else "") + ("p" if bPick else "") + ("t" if bTrunk else "")
        wfeatures = ("h" if wHead else "") + ("p" if wPick else "") + ("t" if wTrunk else "")

        print(f"{bname} ({brating}) vs {wname} ({wrating}): {game.score}, {bcount} ({bfeatures}) / {wcount} ({wfeatures}) recent moves")

    print(f"Dataset checked, {tcount} train/{vcount} validation/{ecount} test, {len(errors)} errors.")
    headinfo = "all" if allHead else "some" if anyHead else "none"
    pickinfo = "all" if allPick else "some" if anyPick else "none"
    trunkinfo = "all" if allTrunk else "some" if anyTrunk else "none"
    print(f"  head features? {headinfo}")
    print(f"  pick features? {pickinfo}")
    print(f"  trunk features? {trunkinfo}")

    for game, message in errors:
        print(f"Error in {game.black.name} vs {game.white.name}: {message}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read a list file (dataset) and print out the marked rows and available features.")
    parser.add_argument("listpath", type=str, help="CSV file listing games and labels")
    parser.add_argument("featuredir", type=str, help="Directory containing extracted features")
    parser.add_argument("--require-head", action="store_true", help="If set, consider games missing head features as errors")
    parser.add_argument("--require-pick", action="store_true", help="If set, consider games missing pick features as errors")
    parser.add_argument("--require-trunk", action="store_true", help="If set, consider games missing trunk features as errors")
    args = parser.parse_args()

    main(args.listpath, args.featuredir, require_head=args.require_head, require_pick=args.require_pick, require_trunk=args.require_trunk)

