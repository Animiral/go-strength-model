This repository contains scripts and material for my strength model based on KataGo, which is the subject of my master thesis.

The strength model is a neural network model which uses the existing KataGo infrastructure and a new additional strength head component to predict players' strength rating from recent moves played. This document gives you step-by-step instructions for training and running the strength model.

# External Resources

The following external dependencies are required:

* the [sgfmill](https://github.com/mattheww/sgfmill) Python package: `pip3 install sgfmill`
* my [fork of the katago repository](https://github.com/Animiral/KataGo), originally [here](https://github.com/lightvector/KataGo)
* any (KataGo network)[https://katagotraining.org/networks/]
* my [fork of the goratings repository](https://github.com/Animiral/goratings), originally [here](https://github.com/online-go/goratings)
* a dataset to work on, like the [OGS 2021 collection](https://archive.org/details/ogs2021)
* a CUDA compatible graphics card, because the modified KataGo is currently restricted to the CUDA backend

# Dataset Preparation

We start by preparing the games which we want to use in training. We assume that these games exist as a collection of SGF files found under some common root directory on your disk.

## Filtering Games

The `sgffilter.py` script provided in this repository traverses a given directory and all its subdirectories for SGF files. Every file that contains a suitable training game is printed to the output file. Suitable games are no-handicap even 19x19 games with more than 5 seconds per move to think, have at least 20 moves played, were decided by either counting, resignation or timeout, and contain the string "ranked" (and not "unranked") in the GC property.

```
$ python3 sgffilter.py path/to/dataset more/paths/to/datasets --output games.csv
```

## Judging Games

In this optional step, we override the specified winner of each game in the list with whoever held the advantage at the end in the eyes of KataGo. The goal is to improve the quality of the training data. In reality, games are often won by the player in the worse position. This can happen if their time runs out, if they feel lost and resign, or especially among beginners, the game reaches the counting stage and is scored wrong by the players. By eliminating these factors, we concentrate on the effectiveness of the moves played.

The forked KataGo repository contains the script `judge_gameset.py`, which can read our prepared `games.csv` and output a new list with predicted winners.

```
$ python3 path/to/katago/python/judge_gameset.py -katago-path path/to/katago/cpp/katago -config-path path/to/katago/cpp/configs/analysis_example.cfg -model-path path/to/model.bin.gz -i games.csv -o games_judged.csv
```

## Glicko2 Calculation

We feed our dataset(s) into our reference rating algorithm Glicko2, which is implemented for OGS in the goratings repository. It contains the script `analyze_glicko2_one_game_at_a_time.py`. The forked repository is extended to read input from our games list and SGF files, and to produce an output list that contains the results of the rating calculation after every game.

```
$ GORATINGS_DIR=path/to/goratings
$ PYTHONPATH="$PYTHONPATH:$GORATINGS_DIR" python3 $GORATINGS_DIR/analysis/analyze_glicko2_one_game_at_a_time.py \
	--sgf games_judged.csv --analysis-outfile games_glicko_ids.csv --mass-timeout-rule false
```

Since the scripts in goratings use integer IDs for games and players, we need to run our `name_ratings.py` script to restore SGF paths and player names.

```
$ python3 name_ratings.py --list games_judged.csv --ratings games_glicko_ids.csv --output games_glicko.csv
```

## Strength Model Calculation

Once the strength model is trained, we can apply it to a dataset by invoking the modified KataGo (needs to be compiled from my fork, see above) with the `rating_system` command.

```
$ KATA_MODEL=path/to/model.bin.gz
$ STRENGTH_MODEL=path/to/strengthmodel.bin.gz
$ CONFIG=configs/analysis_example.cfg
$ LISTFILE=games_judged.csv
$ OUTFILE=games_strmodel.csv
$ FEATUREDIR=path/to/featurecache
$ katago rating_system -model $KATA_MODEL -strengthmodel $STRENGTH_MODEL -config $CONFIG -list $LISTFILE -outlist $OUTFILE -featuredir $FEATUREDIR
```

The `-featuredir` is optional, but if we expect to run this command more than once, then the extracted move features will be dumped in this directory, where they can be quickly retrieved in the future. Move features are the outputs of the KataGo network and inputs to the strength model. Since the KataGo network is static for us, while the strength model is trained, it saves time to do this expensive precomputation just once.

The output file contains the results of the rating calculation, directly comparable to the output of the Glicko2 analysis script above.

## Rating the Rating System

The quality of a rating system is measured by its ability to predict the winners of matchups as they happen. When the higher-rated player beats the lower-rated player, the system was successful. Moreover, we value not just the number of successfully predicted matchups, but also the degree of the prediction. The higher the prior rating of the eventual winner compared to the the loser, the more performant our system.
We measure the success rate as the number of successful predictions divided by the total number of matches. We measure the performance as the sum of log-likelihoods of every outcome prediction (logp). This is the total log-likelihood of all the outcomes happening as they did according to the strength model based on the prior information at the time.

Given a rating calculation file like `games_glicko.csv` or `games_strmodel.csv` in the examples above, that contains the Winner and WhiteWinrate of every game, the script `calc_performance.py` tells us the success rate and log-likelihood achieved by the system. It also counts with and without games involving new players, who come into the system with no prior information.

```
$ python3 calc_performance.py games_strmodel.csv
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
$ python3 label_gameset.py --list games_glicko.csv --output games_labels.csv --advance 10
```

# Training

Using the dataset as prepared above, we can train the strength model on it – either from scratch, or by loading an existing model file.
The strength model is implemented as a modification to KataGo, the C++ program. Note that KataGo, apart from its main program, also consists of Python scripts which are used to train the KataGo model itself. We disregard these training programs, as our training is implemented entirely in C++.

## The Training Command

The modified KataGo version from my fork (see above) implements the new `strength_training` command. Invoke it from the shell like this:

```
$ KATAGO=path/to/katago
$ KATA_MODEL=path/to/model.bin.gz
$ STRENGTH_MODEL=path/to/strengthmodel.bin.gz
$ CONFIG=configs/strength_analysis_example.cfg
$ LISTFILE=games_labels.csv
$ FEATUREDIR=path/to/featurecache
$ katago strength_training -model $KATA_MODEL -strengthmodel $STRENGTH_MODEL -config $CONFIG -list $LISTFILE -featuredir $FEATUREDIR
```

Please keep in mind that relative SGF paths in `LISTFILE` must be relative to the current working directory.

Currently, the model is a simple proof of concept. After training completes, the result is saved in the file given as `STRENGTH_MODEL`.
