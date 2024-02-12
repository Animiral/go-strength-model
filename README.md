This repository contains scripts, utilities and material for my strength model based on KataGo, which is the subject of my master thesis.

The strength model is a neural network model which uses the existing KataGo infrastructure and a new additional strength head component to predict players' strength rating from recent moves played. This document gives you step-by-step instructions for training and running the strength model.

Python scripts are located in the `python` subdirectory. By convention, we dump CSV files in the `csv` subdirectory, which you may need to create first.

# External Resources

The following external dependencies are required:

* my [fork of the katago repository](https://github.com/Animiral/KataGo), originally [here](https://github.com/lightvector/KataGo)
* any [KataGo network](https://katagotraining.org/networks/)
* a CUDA compatible graphics card, because the modified KataGo is currently restricted to the CUDA backend
* to use the strength model: a fully trained strength model file
* to train the strength model from scratch:
  - the [sgfmill](https://github.com/mattheww/sgfmill) Python package: `pip3 install sgfmill`
  - my [fork of the goratings repository](https://github.com/Animiral/goratings), originally [here](https://github.com/online-go/goratings)
  - a dataset to work on, like the [OGS 2021 collection](https://archive.org/details/ogs2021)

# Estimate Playing Strength

Using the modified KataGo and a strength model file, we can let the program estimate a player's Glicko rating.
The strength model file can be obtained, for example, by following the further steps in this README to train it on an existing game dataset.
In addition to that model file, we only need to pass it a set of SGF files and a player name.

```
$ KATAGO=path/to/katago/cpp/katago
$ CONFIG=path/to/katago/cpp/configs/analysis_example.cfg
$ MODEL=path/to/katago/models/kata1-b18c384nbt-s6582191360-d3422816034.bin.gz
$ STRENGTH_MODEL=$KATADIR/models/strength-model.bin
$ PLAYER=playername
$ SGF=sgf/game1.sgf sgf/game2.sgf sgf/game3.sgf
$ $KATAGO strength_analysis -config $CONFIG -model $MODEL -strengthmodel $STRENGTH_MODEL -player $PLAYER $SGF
```

# Dataset Preparation

We start by preparing the games which we want to use in training. We assume that these games exist as a collection of SGF files found under some common root directory on your disk.

## Filtering Games

The `sgffilter.py` script provided in this repository traverses a given directory and all its subdirectories for SGF files. Every file that contains a suitable training game is printed to the output file. Suitable games are no-handicap even 19x19 games with more than 5 seconds per move to think, have at least 20 moves played, were decided by either counting, resignation or timeout, and contain the string "ranked" (and not "unranked") in the GC property.

```
$ python3 python/sgffilter.py path/to/dataset more/paths/to/datasets --output csv/games.csv
```

## Judging Games

In this optional step, we override the specified winner of each game in the list with whoever held the advantage at the end in the eyes of KataGo. The goal is to improve the quality of the training data. In reality, games are often won by the player in the worse position. This can happen if their time runs out, if they feel lost and resign, or especially among beginners, the game reaches the counting stage and is scored wrong by the players. By eliminating these factors, we concentrate on the effectiveness of the moves played.

The forked KataGo repository contains the script `judge_gameset.py`, which can read our prepared `games.csv` and output a new list with predicted winners.

```
$ KATAGO=path/to/katago/cpp/katago
$ CONFIG=path/to/katago/cpp/configs/analysis_example.cfg
$ MODEL=path/to/katago/models/kata1-b18c384nbt-s6582191360-d3422816034.bin.gz
$ LIST=csv/games.csv
$ OUTLIST=csv/games_judged.csv
$ FLAGS=""  # FLAGS=--keep-undecided --max-visits 50
$ python3 ~/source/katago/python/judge_gameset.py $FLAGS --katago-path $KATAGO --config-path $CONFIG --model-path $MODEL -i $LIST -o $OUTLIST
```

The script copies all columns from the input except for `Winner`. The new winner is noted in the `Score` column of the output file, with a value of `1` if black wins, `0` if white wins, and `0.5` if the game cannot be decided. Undecided games are omitted from the output unless you pass the flag `--keep-undecided` to the script. The depth of evaluation can be modified with the `--max-visits` argument, which passes through to KataGo.

## Splitting the Dataset

The script `random_split.py` reads a CSV file and adds or modifies the "Set" column, which marks each row as a member in one of three sets: "T" for the *training set*, "V" for the *validation set* and "E" for the *test set*. The markers are distributed randomly with prevalence according to user-defined "fraction" parameters.

The motivation behind assigning rows to sets instead of splitting the entire match pool is that if we just form distinct pools from the original one, we tear apart player's rating histories, depriving our algorithms of the data from which they derive their predictions. Instead, we keep them in the same pool. In the training process, we train only on training matches and test only on test matches, while the combined match data is available in the rating history. This technique stems from link prediction problems in social networks, where random test edges are removed from the full graph and later predicted by the model trained on the remaining edges.

Run the set assignment script as follows.

```
$ python3 python/random_split.py --input csv/games_judged.csv --output csv/games_judged.csv --trainingFraction 0.8 --validationFraction 0.1
```

This will allocate 80% of all rows to the training set, 10% to the validation set and the remaining 10% to the test set.

Once allocated, the script can also copy the same set markers to a different CSV file, as long as the "copy-from" file has both "File" and "Set" headers and holds the information on every "File" listed in the input CSV file:

```
$ python3 python/random_split.py --input csv/games_judged.csv --copy csv/games_labels.csv
```

## Glicko-2 Calculation

We feed our dataset(s) into our reference rating algorithm Glicko-2, which is implemented for OGS in the goratings repository. It contains the script `analyze_glicko2_one_game_at_a_time.py`. The forked repository is extended to read input from our games list and SGF files, and to produce an output list that contains the results of the rating calculation after every game.
From our list file, the script expects the file name in the first column and the score in the last column. This is the case in `games_judged.csv` from above (before adding the set marker!).

```
$ GORATINGS_DIR=path/to/goratings
$ PYTHONPATH="$PYTHONPATH:$GORATINGS_DIR" python3 $GORATINGS_DIR/analysis/analyze_glicko2_one_game_at_a_time.py \
	--sgf csv/games_judged.csv --analysis-outfile csv/games_glicko_ids.csv --mass-timeout-rule false
```

Since the scripts in goratings use integer IDs for games and players, we need to run our `name_ratings.py` script to restore SGF paths and player names.

```
$ python3 python/name_ratings.py --list csv/games_judged.csv --ratings csv/games_glicko_ids.csv --output csv/games_glicko.csv
```

This step is “dataset preparation” in the sense that we may train our model on future Glicko ratings, see Training section below. Otherwise, Glicko-2 is a reference rating system for us.

## Move Feature Precomputation

Move features are the outputs of the KataGo network and inputs to the strength model. Since the KataGo network is static for us, while the strength model is trained, it saves time to do this expensive precomputation just once. Launch the command to precompute all features for every game in a list file and write them to feature files in a dedicated directory as follows:

```
$ KATA_MODEL=path/to/model.bin.gz
$ FEATUREDIR=path/to/featurecache
$ katago extract_features -model $KATA_MODEL -config $CONFIG -list $LISTFILE -featuredir $FEATUREDIR
```

# Dataset Viewer

`datasetviewer` is a utility program that allows us to query data from the dataset. It must be compiled from its C++ sources, located in this repository in the `datasetviewer` subdirectory.

```
$ cd datasetviewer
$ cmake .
$ make
```

Start the program with the dataset list file (generated following the previous sections) and feature cache directory as arguments. If those arguments are not provided, `datasetviewer` explains itself.

```
Usage: datasetviewer LIST_FILE FEATURE_DIR
  View data from the games in the LIST_FILE, with precomputed features stored in FEATURE_DIR.
Commands:
  help                                  Print this help.
  info                                  Print active settings and filters.
  exit                                  Exit program.
  select TOPIC FILTER OP VALUE|RANGE    Set the filter for TOPIC.
    TOPIC choices: games|moves
    FILTER choices for games: none|#|file|black|white|score|predscore|set
    FILTER choices for moves: none|recent|color
    OP choices: in|contains
  configure SETTING VALUE               Configure a global setting.
    SETTING choices: window
  print TOPIC COLUMN...                 Write the values to stdout.
  dump FILE TOPIC COLUMN...             Write the values to FILE.
    TOPIC choices: games|moves
    COLUMN choices for games: #|file|black.name|white.name|black.rating|white.rating|black.rank|white.rank|score|predscore|set
    COLUMN choices for moves: #|color|winprob|lead|policy|maxpolicy|wrloss|ploss|rating
```

For example, in the following session, we extract the first 100 matchups, and also recent move data for game 11 in the dataset.

```
$ VIEWERDIR=datasetviewer
$ LIST=csv/games_labels.csv
$ FEATUREDIR=featurecache
$ $VIEWERDIR/datasetviewer $LIST $FEATUREDIR
Dataset Viewer: 1890 games read from games_labels.csv (with features), ready.
> select games # in 0-99
Ok.
> dump matches_100.csv games black.name black.rating white.name white.rating
Write to matches_100.csv...
Done.
> select moves recent in 11
Ok.
> dump recent_11.csv moves color policy ploss
Write to recent_11.csv...
Done.
> exit 
Dataset Viewer: bye!
$ 
```

# Training

Using the dataset as prepared above, we can train the strength model on it – either from scratch, or by loading an existing model file.
The strength model is implemented as a modification to KataGo, the C++ program. Note that KataGo, apart from its main program, also consists of Python scripts which are used to train the KataGo model itself. We disregard these training programs, as our training is implemented entirely in C++.

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
$ python3 python/label_gameset.py --list csv/games_glicko.csv --output csv/games_labels.csv --advance 10
```

## The Training Command

The modified KataGo version from my fork (see above) implements the new `strength_training` command. Invoke it from the shell like this:

```
$ KATAGO=path/to/katago
$ STRENGTH_MODEL=path/to/strengthmodel.bin.gz
$ CONFIG=configs/strength_analysis_example.cfg
$ LISTFILE=csv/games_labels.csv
$ FEATUREDIR=path/to/featurecache
$ katago strength_training -strengthmodel $STRENGTH_MODEL -config $CONFIG -list $LISTFILE -featuredir $FEATUREDIR
```

Please keep in mind that relative SGF paths in `LISTFILE` must be relative to the current working directory.
If the `LISTFILE` contains the "Set" column from the previous step, the matches will be used according to their designation. The program trains the model on training matches, reporting progress on the training and validation sets after every epoch.

Currently, the model is a simple proof of concept. After training completes, the result is saved in the file given as `STRENGTH_MODEL`.

# Evaluation

Given a CSV rating calculation file with the required columns, our script `calc_performance.py` calculates the relevant metrics of the rating system which produced the input file.
The required columns are:

* `Score` and `PredictedScore`
* `Player Black` and `Player White`, to distinguish between first-timers and players with information attached
* `Set` (optional), to calculate only on set `T`, `V` or `E` (e.g. `V` for validation set)

These columns are present in the files produced in the relevant sections: `games_glicko.csv` from “Glicko-2 Calculation” above, and the outputs of the following steps.

## Stochastic Model Calculation

The Stochastic Model is a simple idea that we can predict winning chances based on the expected points loss of both players in their match.
It is implemented in the modified KataGo (needs to be compiled from my fork, see above) with the `rating_system` command.

```
$ CONFIG=configs/analysis_example.cfg
$ LISTFILE=csv/games_judged.csv
$ OUTFILE=csv/games_stochastic.csv
$ FEATUREDIR=path/to/featurecache
$ katago rating_system -config $CONFIG -list $LISTFILE -outlist $OUTFILE -featuredir $FEATUREDIR -set V
```

The `-featuredir` is mandatory and must hold the precomputed extracted move features for every game. These must be prepared by `extract_features` as outlined above.

The output file contains the results of the rating calculation, directly comparable to the output of the Glicko-2 analysis script above.

## Strength Model Calculation

Once the strength model is trained, we can apply it to a dataset by invoking modified KataGo as above, with the `-strengthmodel` parameter:

```
$ STRENGTH_MODEL=path/to/strengthmodel.bin.gz
$ CONFIG=configs/analysis_example.cfg
$ LISTFILE=csv/games_judged.csv
$ OUTFILE=csv/games_strmodel.csv
$ FEATUREDIR=path/to/featurecache
$ katago rating_system -strengthmodel $STRENGTH_MODEL -config $CONFIG -list $LISTFILE -outlist $OUTFILE -featuredir $FEATUREDIR -set V
```

The `-featuredir` is again mandatory and the output file is a valid rating calculation file.

## Rating the Rating Systems

The quality of a rating system is measured by its ability to predict the winners of matchups as they happen. When the higher-rated player beats the lower-rated player, the system was successful. Moreover, we value not just the number of successfully predicted matchups, but also the degree of the prediction. The higher the prior rating of the eventual winner compared to the the loser, the more performant our system.
We measure the success rate as the number of successful predictions divided by the total number of matches. We measure the performance as the average of log-likelihoods of every outcome prediction (logp). This is the log-likelihood of all the outcomes happening as they did according to the rating system based on the prior information at the time, scaled with dataset size for better comparison.

Given a rating calculation file as defined above, the script `calc_performance.py` tells us the success rate and log-likelihood achieved by the system. It also counts with and without games involving new players, who come into the system with no prior information.

```
$ python3 python/calc_performance.py csv/games_strmodel.csv -m V
```

# Tests

The modified katago features new tests for the new functionality.

```
$ katago runstrengthmodeltests
```

# Plots

Visual presentations of the data found in the thesis are created using scripts in the `plots` subdirectory. Consult [the associated HowTo](plots/HOWTO.md) for reproduction steps.
