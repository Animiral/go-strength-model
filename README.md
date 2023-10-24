This repository contains scripts and material for my strength model based on KataGo, which is the subject of my master thesis.

The strength model is a neural network model which uses the existing KataGo infrastructure and a new additional strength head component to predict players' strength rating from recent moves played. This document gives you step-by-step instructions for training and running the strength model.

# External Resources

The following external dependencies are required:

* the [sgfmill](https://github.com/mattheww/sgfmill) Python package: `pip3 install sgfmill`
* my fork of the katago repository, originally at https://github.com/lightvector/KataGo
* my fork of the goratings repository, originally at https://github.com/online-go/goratings
* a dataset to train on, like the [OGS 2021 collection](https://archive.org/details/ogs2021)

# Dataset Preparation

We start by preparing the games which we want to use in training. We assume that these games exist as a collection of SGF files found under some common root directory on your disk.

## Filtering Games

The `sgffilter.py` script provided in this repository traverses a given directory and all its subdirectories for SGF files. Every file that contains a suitable training game is printed to the output file. Suitable games are no-handicap even 19x19 games with more than 5 seconds per move to think, have at least 20 moves played, were decided by either counting, resignation or timeout, and contain the string "ranked" (and not "unranked") in the GC property.

```
python3 sgffilter.py path/to/dataset more/paths/to/datasets --output games.csv
```

## Judging Games

In this optional step, we override the specified winner of each game in the list with whoever held the advantage at the end in the eyes of KataGo. The goal is to improve the quality of the training data. In reality, games are often won by the player in the worse position. This can happen if their time runs out, if they feel lost and resign, or especially among beginners, the game reaches the counting stage and is scored wrong by the players. By eliminating these factors, we concentrate on the effectiveness of the moves played.

The forked KataGo repository contains the script `judge_gameset.py`, which can read our prepared `games.csv` and output a new list with predicted winners.

```
python3 path/to/katago/python/judge_gameset.py -katago-path path/to/katago/cpp/katago -config-path path/to/katago/cpp/configs/analysis_example.cfg -model-path path/to/model.bin.gz -i games.csv -o games_judged.csv
```

## Glicko2 Calculation

We feed our training set into our reference rating algorithm Glicko2, which is implemented for OGS in the goratings repository. It contains the script `analyze_glicko2_one_game_at_a_time.py`. The forked repository is extended to read input from our games list and SGF files, and to produce an output list that contains the results of the rating calculation after every game.

```
GORATINGS_DIR=path/to/goratings
PYTHONPATH="$PYTHONPATH:$GORATINGS_DIR" python3 $GORATINGS_DIR/analysis/analyze_glicko2_one_game_at_a_time.py \
	--sgf games_judged.csv --analysis-outfile games_glicko_ids.csv --mass-timeout-rule false
```

Since the scripts in goratings use integer IDs for games and players, we need to run our `name_ratings.py` script to restore SGF paths and player names.

```
python3 name_ratings.py --list games_judged.csv --ratings games_glicko_ids.csv --output games_glicko.csv
```

## Labeling Games

One way to train our strength model is to let it predict the players' future rating number. The `label_gameset.py` script provided in this repository reads the list of games that we produced in the previous steps. The output games list contains the future rating of both the black and the white player involved, from the point when they have played an additional number of games as specified in the `--advance` argument.

For example, if Alice starts with a rating of 1000 and then plays against B, C, D, E and F, resulting in ratings of 1100, 1200, 1300, 1400 and 1500 respectively, and the `--advance` option is set to 3 (games in the future), then the resultant labeling might be:

```
File,Player White,Player Black,Winner,Label White,Label Black
Alice_vs_B.sgf,Alice,B,W+,1300,1000
Alice_vs_C.sgf,Alice,C,W+,1400,1000
Alice_vs_D.sgf,Alice,D,W+,1500,1000
Alice_vs_E.sgf,Alice,E,W+,1500,1000
Alice_vs_F.sgf,Alice,F,W+,1500,1000
```

Run the script as follows.

```
python3 label_gameset.py --list games_glicko.csv --output games_labels.csv --advance 10
```
