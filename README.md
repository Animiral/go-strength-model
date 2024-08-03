This repository contains scripts, utilities and material for my strength model based on KataGo, which is the subject of my master thesis.

The strength model is a neural network model which uses the existing KataGo network to interpret Go positions and moves. It uses the internal result representation of KataGo as input in its own architecture to predict players' strength rating from recent moves played. This document gives you step-by-step instructions for training and running the strength model.

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

Using the modified KataGo and a strength model weights file, we can estimate a player's Glicko rating.
The strength model weights file can be obtained, for example, by following the further steps in this README to train it on an existing game dataset.

The strength model is implemented in Python using PyTorch, but it requires the modified katago binary to extract move features from the SGFs.
To pass all the required and optional arguments, follow this script.

```
$ SGF=path/to/games/*.sgf
$ KATAGO=path/to/katago/cpp/katago
$ KATAMODEL=path/to/katago/models/kata1-b18c384nbt-s6582191360-d3422816034.bin.gz
$ KATACONFIG=path/to/katago/cpp/configs/analysis_example.cfg
$ STRMODEL=path/to/weights/file.pth
$ FEATURENAME=pick  # or trunk, or head, if compatible with model
$ PLAYERNAMEARG=--playername \"My Name\"  # needs to match player name in SGFs
#$ PLAYERNAMEARG=   # uncomment this to auto-detect name

python3 python/model/run.py $SGF --katago $KATAGO --katamodel $KATAMODEL --kataconfig $KATACONFIG \
    --model $STRMODEL --featurename $FEATURENAME $PLAYERNAMEARG
```

# Dataset Preparation

We start by preparing the games which we want to use in training. We assume that these games exist as a collection of SGF files found under some common root directory on your disk.

## Filtering Games

The `extractor` program provided in this repository searches through an archive of SGF files for suitable training games. Every eligible file is extracted to the dataset directory to a file path constructed from the game date specified in the SGF and the names of the players. Additionally, all SGF paths are printed to a CSV output file. Suitable games are no-handicap even 19x19 games with more than 5 seconds per move to think, have at least 20 moves played, no passes before move 50, were decided by either counting, resignation or timeout, and contain the string "ranked" (and not "unranked") in the GC property.

The `extractor` must be compiled from its C++ sources, located in this repository in the `extractor` subdirectory.

```
$ pushd extractor
$ cmake .
$ make
$ popd
```

Start the program with the archive file (containing SGFs), extraction base directory path and CSV output path as arguments. If those arguments are not provided, `extractor` explains itself.

```
$ mkdir dataset
$ extractor/extractor sgfs.tar.gz dataset csv/games.csv
```

As an alternative, this project also offers a script that just builds the CSV file from all eligible SGFs in a given directory and subdirectories, and another utility to filter a game list in CSV format. See "Filtering Games (alternative)" section. The advantage of `extractor` is that it is fast and there is no need to extract a large dataset, including undesirable SGFs, to disk. Beyond that, `extractor` extracts everything using file names with characters `[a-zA-Z0-9_-. ]` only, for better compatibility even if the players' names include characters not allowed in the target filesystem. On top of that, it features hacks to properly read some broken player names from the OGS 2021 dataset specifically.
The `namecompat` utility bundled in the `extractor` directory can perform just the name extraction with corrections as its own step.

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
If the process of judging games should be interrupted, the script can resume from any point. If the output file exists prior to running, all SGF names in it are excluded from being run through KataGo and new results are appended to the file.

For a dataset of 7M+ games, even with an optimized KataGo build with TensorRT backend, using a `b18c384nbt` network and `--max-visits 10`, running on an RTX 4070 GPU, this can take over 40 hours total to run.

## Splitting the Dataset

The script `random_split.py` reads a CSV file and adds or modifies the "Set" column, which marks a number of rows as a member in one of three sets: "T" for the *training set*, "V" for the *validation set*, "E" for the *test set* and "X" for the *exhibition set*. Rows not in any set are marked with "-". The markers are distributed randomly, either as a proportion relative to the whole dataset if the user-defined "part" parameters are <1, or to an absolute number of rows given in the parameters if they are >=1.

The motivation behind assigning rows to sets instead of splitting the entire match pool is that if we just form distinct pools from the original one, we tear apart player's rating histories, depriving our algorithms of the data from which they derive their predictions. Instead, we keep them in the same pool. In the training process, we train only on training matches and test only on test matches, while the combined match data is available in the rating history. This technique stems from link prediction problems in social networks, where random test edges are removed from the full graph and later predicted by the model trained on the remaining edges.

Run the set assignment script as follows.

```
$ python3 python/random_split.py --input csv/games_judged.csv --output csv/games_judged.csv --trainingPart 10000 --validationPart 5000 --testPart 5000 --exhibitionPart 5000
```

This will allocate 10000 rows to the training set, 5000 to the validation set, 5000 to the test set and 5000 to the exhibition set. Any remaining rows are left unassigned, but still part of the dataset, forming the players' histories and acting as a source of recent moves. Just the model will not be trained or tested on these data points. Because all games with a set marker (more specifically, their recent move sets) must be preprocessed through the KataGo network, it is not feasible to mark millions of games for training.

If not specified, the `--output` file defaults to the same as the `--input`, overwriting it with the added information.

Rows that introduce a specific player for the first time in the dataset are generally not eligible for marking as any set, because these rows do not offer the necessary prior information for a model to predict the match outcome. The optional `--with-novice` switch disables this behavior, making all rows eligible for inclusion in one of the sets.

The script can also check for noisy rows and not mark these as training rows. Noisy rows are rows where both players do not have a minimum number of future games in the dataset, meaning that their label does not have future information from the Glicko-2 system and might be less accurate. The number of required future games is specified with the optional `--advance` parameter, just like in the section "Labeling Games" below. A row is also noisy if the labels disagree with the score (outcome). I.e. black wins against higher-rated white or vice-versa. The noise criteria should only be applied to a labeled dataset (see below) and thus are only used if `--advance` is specified.

A row qualifies to be in the exhibition set if neither the black nor the white player have more than 4 games of past history at the point of the match.

With the optional `--modify` switch, the existing set assignment will be kept as far as possible, keeping changes to a minimum.

As an alternative usage, the splits can be specified as fractions. Omitting `--testPart` assigns all remaining rows to the test set.

```
$ python3 python/random_split.py --input csv/games_judged.csv --output csv/games_judged.csv --trainingPart 0.8 --validationPart 0.1
```

This will allocate 80% of eligible rows to the training set, 10% to the validation set and the remaining 10% to the test set.

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

We make use of Glicko-2 ratings in two ways.
First, it provides the after-game ratings `BlackRating` and `WhiteRating` as the basis for future Glicko ratings labels, which our models are trained and evaluated on, see Training section below.
Second, it is our reference rating system. The `PredictedScore`, `PredictedBlackRating` and `PredictedWhiteRating` columns allow us to measure the performance of Glicko-2 itself on our dataset.

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
The strength model is implemented in Python using PyTorch. It requires that the input game(s) are preprocessed through the KataGo network to obtain the strength model input features for every move. TODO: unify this process in one script

## Labeling Games

One way to train our strength model is to let it predict the players' future rating number. The `label_gameset.py` script provided in this repository reads the list of games that we produced in the previous steps. The output games list contains the future rating of both the black and the white player involved, from the point when they have played an additional number of games as specified in the `--advance` argument.

For example, if Alice starts with a rating of 1000 and then wins against B, C, D, E and F, resulting in ratings of 1100, 1200, 1300, 1400 and 1500 respectively, and the `--advance` option is set to 3 (games in the future), then the resultant labeling might be:

```
File,Player White,Player Black,Score,WhiteRating,BlackRating
Alice_vs_B.sgf,Alice,B,0,1300,1000
Alice_vs_C.sgf,Alice,C,0,1400,1000
Alice_vs_D.sgf,Alice,D,0,1500,1000
E_vs_Alice.sgf,E,Alice,1,1000,1500
Alice_vs_F.sgf,Alice,F,0,1500,1000
```

Run the script as follows.

```
$ python3 python/label_gameset.py --list csv/games_glicko.csv --output csv/games_labels.csv --advance 10
```

## Precomputation

The strength prediction for a player is based on a large number of *recent moves*, every one of which must be evaluated by the KataGo network to find its features.

The precomputation of recent move features is built into my fork of KataGo in the form of the new command `extract_features`. This extractor currently supports three types of features:

* *Head features* are derived from the usual output of the KataGo network. They include *winrate loss* and *points loss*, which require evaluating both the position before and after the move in question. We use these features in our basic stochastic model and as data to train our proof-of-concept model.
* *Trunk features* simply contain the entire trunk output of the KataGo network from the board state of the recent move. They take too much time and generate too much data for our dataset with 20000 marked rows (see *Splitting the Dataset* above).
* *Pick features* contain the feature vector from the trunk output associated with the board location where the stone was placed. We use these features to train our full model.

Invoke the `extract_features` command as follows:

```
$ KATAGO=path/to/katago
$ MODEL=path/to/katago/models/kata1-b18c384nbt-s6582191360-d3422816034.bin.gz
$ CONFIG=path/to/configs/analysis_example.cfg
$ LIST=csv/games_labels.csv
$ FEATUREDIR=path/to/featurecache
$ SELECTION="-with-trunk -with-pick -with-head"  # feature categories to precompute
$ WINDOWSIZE=500   # number of recent moves per game and player
$ BATCHSIZE=10     # number of board positions sent to KataGo in one batch
$ BATCHTHREADS=8   # number of concurrent worker threads launching independent NN batches to the GPU
$ WORKERTHREADS=8  # number of concurrent CPU workers to prepare NN inputs and process NN outputs

$ katago extract_features $SELECTION -model $MODEL -config $CONFIG -list $LIST -featuredir $FEATUREDIR \
                          -window-size $WINDOWSIZE -batch-size $BATCHSIZE -batch-threads $BATCHTHREADS \
                          -worker-threads $WORKERTHREADS
```

KataGo reads the given list file and determines the recent move set for every marked game in the dataset. It uses its network to evaluate all the necessary positions from which it derives the features specified in the selection parameters.
The features associated with the recent moves are grouped by the game in which they occur and the player's color, then written to a zip file on disk stored in the feature cache directory. The file name ends in `Features.zip`.
Once all recent move features are avialable per recent game, it combines them into recent move set archives with file names based on the game and player color that these moves are recent for and stores them in the feature cache directory. The file name ends in `Recent.zip`. These are our data sources for model training.

For example, if game 110 between players P and Q is in the training set, where P has recent moves in games 101, 105 and 107, while Q has recent moves in games 102 and 106, then the extractor creates `Features.zip` archives for games 101, 102, 105, 106 and 107. These combine into one `Recent.zip` archive for each side in game 110.

This tool supports two more command line switches. The `-recompute` switch causes the program to overwrite any files left over from previous runs. Omit it to continue precomputation from the state of the previous, unfinished run. The `-print-recent-moves` switch outputs some debug information (not just recent moves).

This step can be very time and resource intensive, especially with large datasets (multiple 10k marked games), large window size, large batch size and many threads. Try different parameters to find an acceptable balance between speed and resource use on your system. In case of crashes due to resource exhaustion, the process can be resumed without the `-recompute` switch.

After successful completion, check the result using `checkdataset.py` found in this repository:

```
$ LIST=csv/games_labels.csv
$ FEATUREDIR=path/to/featurecache
$ SELECTION="--require-trunk --require-pick --require-head"  # features to check
$ python3 python/checkdataset.py $LIST $FEATUREDIR $SELECTION
```

This script tallies up all marked games in every set and look at their recent move zip files. If required files or features in them are missing or of inconsistent size, the affected game is listed in an error summary at the end. Ideally, the end of its output should look like this, with desired feature sets "all" present:

```
Dataset checked, 10000 train/5000 validation/5000 test, 0 errors.
  head features? all
  pick features? all
  trunk features? none
```

## The Training Command

The `train.py` file included in this repository can be launched as a standalone script to train the strength model on the above precomputed data. Invoke it from the shell like this:

```
LIST=csv/games_labels.csv
FEATUREDIR=path/to/featurecache
FEATURENAME=pick
OUTFILE=nets/model{}.pth
TRAINLOSSFILE=logs/trainloss.txt
VALIDATIONLOSSFILE=logs/validationloss.txt
LOGFILE=logs/traininglog.txt

BATCHSIZE=100
STEPS=100
EPOCHS=100
LEARNINGRATE=0.001
LRDECAY=0.95
PATIENCE=3

WINDOW_SIZE=500
DEPTH=5
HIDDEN_DIMS=64
QUERY_DIMS=64
INDUCING_POINTS=32

python3 -u python/model/train.py $LIST $FEATUREDIR --featurename $FEATURENAME --outfile "$OUTFILE" \
  --trainlossfile $TRAINLOSSFILE --validationlossfile $VALIDATIONLOSSFILE \
  --batch-size $BATCHSIZE --steps $STEPS --epochs $EPOCHS --learningrate $LEARNINGRATE --lrdecay $LRDECAY --patience $PATIENCE \
  --window-size $WINDOW_SIZE --depth $DEPTH --hidden-dims $HIDDEN_DIMS --query-dims $QUERY_DIMS --inducing-points $INDUCING_POINTS
```

Please keep in mind that relative SGF paths in `LISTFILE` must be relative to the current working directory.
The `LISTFILE` must contain the "Set" column from the labeling step. The script uses 'T' (training) rows for training and 'V' (validation) rows to check performance.
After every epoch, the trained network weights are saved in a separate file according to the pattern given as `OUTFILE`. The epoch number takes the place of the placeholder `{}` in the final name.

## Hyperparameter Optimization

The training method as specified in the thesis uses a random search for the best hyperparameters. This process is handled by the script `hpsearch.py` in this repository. Invoke it as follows.

```
LIST=csv/games_labels.csv
FEATUREDIR=path/to/featurecache
FEATURENAME=pick
TITLE=search
NETDIR=nets
LOGDIR=logs
BATCHSIZE=100
STEPS=100
EPOCHS=100
PATIENCE=3
SAMPLES=15
BROADITERATIONS=2
FINEITERATIONS=2

python3 -u python/model/hpsearch.py $LIST $FEATUREDIR --featurename $FEATURENAME --title "$TITLE" \
  --netdir $NETDIR --logdir $LOGDIR \
  --batch-size $BATCHSIZE --steps $STEPS --epochs $EPOCHS --patience $PATIENCE \
  --samples $SAMPLES --broad-iterations $BROADITERATIONS --fine-iterations $FINEITERATIONS
```

The same notes regarding `LISTFILE` apply as in the Training Command section above.

# Evaluation

Given a CSV rating calculation file with the required columns, our script `calc_performance.py` calculates the relevant metrics of the rating system which produced the input file.
The required columns are:

* `Score` and `PredictedScore`
* `Player Black` and `Player White`, to distinguish between first-timers and players with information attached
* `Set` (optional), to calculate only on set `T`, `V` or `E` (e.g. `V` for validation set)

The primary measurable quality of a rating system is determined by its ability to predict the winners of matchups as they happen. When the higher-rated player beats the lower-rated player, the system was successful. Moreover, we value not just the number of successfully predicted matchups, but also the degree of the prediction. The higher the prior rating of the eventual winner compared to the the loser, the more performant our system.
We measure the success rate as the number of successful predictions divided by the total number of matches. We measure the performance as the average of log-likelihoods of every outcome prediction (logp). This is the log-likelihood of all the outcomes happening as they did according to the rating system based on the prior information at the time, scaled with dataset size for better comparison.

Given a rating calculation file with the above columns, the script `calc_performance.py` tells us the success rate and log-likelihood achieved by the system.

```
$ python3 python/calc_performance.py csv/games_glicko.csv -m V
```

In this example, we evaluate the output file of the Glicko-2 Calculation above, establishing the performance of our reference system. In the following sections, we obtain evaluation files for our own models.

## Stochastic Model Calculation

The Stochastic Model is a simple idea that we can predict winning chances based on the expected points loss of both players in their match.
It is implemented in the script `stochasticmodel.py`.

```
$ LISTFILE=csv/games_labels.csv
$ FEATUREDIR=path/to/featurecache
$ MARKER=V
$ OUTFILE=csv/games_stochastic_$MARKER.csv
$ python3 python/stochasticmodel.py $LISTFILE $FEATUREDIR -m $MARKER $OUTFILE
```

This model requires precomputed head features for all marked records.
The output file contains the predicted game outcomes for feeding into the performance calculation script.

## Strength Model Calculation

Once the strength model is trained, we can apply it to a dataset by invoking the script `eval.py`.

```
$ LIST=csv/games_labels.csv
$ FEATUREDIR=path/to/featurecache
$ MODELFILE=path/to/strengthmodel.pth
$ OUTFILE=csv/games_strmodel.csv

python3 python/model/eval.py "$LIST" "$FEATUREDIR" "$MODELFILE" --outfile "$OUTFILE" --setmarker V
```

The output file contains the games from the list that match the given set marker, extended by new columns for predicted ratings and score.

# Plots

Visual presentations of the data found in the thesis are created using scripts in the `plots` subdirectory. Consult [the associated HowTo](plots/HOWTO.md) for reproduction steps.

# Miscellaneous

## Filtering Games (alternative)

The `sgffilter.py` script provided in this repository traverses a given directory and all its subdirectories for SGF files. Every file that contains a suitable training game is printed to the output file. Suitable games are no-handicap even 19x19 games with more than 5 seconds per move to think, have at least 20 moves played, were decided by either counting, resignation or timeout, and contain the string "ranked" (and not "unranked") in the GC property.

```
$ python3 python/sgffilter.py path/to/dataset more/paths/to/datasets --output csv/games.csv
```

Another alternative is the C++ utility `sgffilter` in the `extractor` directory. It takes an input CSV list of games like `games.csv` and writes to an output CSV list all those games which meet even stricter filter criteria: games with passes or nowhere-moves before move 50 are discarded.

```
$ pushd extractor
$ cmake .
$ make
$ popd
$ extractor/sgffilter csv/games.csv csv/games_refiltered.csv
```

## Recent Moves Precomputation

This script precomputes, for every game in the dataset, for both the black and white side, which games contain their "recent moves". These are the moves that the strength model may use to predict the outcome of that game. We want to use the training, validation and test sets, so we have to run the script for each of them.

```
$ LISTFILE=csv/games_judged.csv
$ FEATUREDIR=path/to/featurecache
$ python3 python/recentmoves.py "$LISTFILE" "$FEATUREDIR" --marker T
$ python3 python/recentmoves.py "$LISTFILE" "$FEATUREDIR" --marker V
$ python3 python/recentmoves.py "$LISTFILE" "$FEATUREDIR" --marker E
```

## Estimate Playing Strength using the C++ Model (obsolete)

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

## Move Feature Precomputation for Proof of Concept Model (obsolete version)

This smaller strength model uses just six features for each move computed from the outputs of the KataGo search engine. These are winrate after move, points lead after move, move policy, max policy on board, winrate loss and points loss. Here the KataGo network is static for us, so it saves time to do this expensive precomputation just once before training the strength model. Launch the command to precompute all features for every game in a list file and write them to feature files in a dedicated directory as follows:

```
$ KATA_MODEL=path/to/model.bin.gz
$ FEATUREDIR=path/to/featurecache
$ katago extract_pocfeatures -model $KATA_MODEL -config $CONFIG -list $LISTFILE -featuredir $FEATUREDIR
```

## Training the C++ Proof of Concept Model

This implementation of the proof of concept strength model is obsolete. It can be trained using the modified KataGo version from my fork (see above). It implements the new `strength_training` command. Invoke it from the shell like this:

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

After training completes, the result is saved in the file given as `STRENGTH_MODEL`.

## Calculation by the C++ Proof of Concept Model

This implementation of the proof of concept strength model is obsolete. However, with such a trained model, we can apply it to a dataset by invoking modified KataGo with the `-strengthmodel` parameter:

```
$ STRENGTH_MODEL=path/to/strengthmodel.bin.gz
$ CONFIG=configs/analysis_example.cfg
$ LISTFILE=csv/games_judged.csv
$ OUTFILE=csv/games_strmodel.csv
$ FEATUREDIR=path/to/featurecache
$ katago rating_system -strengthmodel $STRENGTH_MODEL -config $CONFIG -list $LISTFILE -outlist $OUTFILE -featuredir $FEATUREDIR -set V
```

The `-featuredir` is again mandatory and the output file is a valid rating calculation file.

## Stochastic Model Calculation Implemented in C++

TThis implementation of the Stochastic model is obsolete. The Stochastic Model is a simple idea that we can predict winning chances based on the expected points loss of both players in their match.
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

## C++ Tests

The modified katago features new tests for the new functionality.

```
$ katago runstrengthmodeltests
```
