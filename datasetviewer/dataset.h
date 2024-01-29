#ifndef DATASET_H_
#define DATASET_H_

#include <string>
#include <vector>
#include <map>

// this is what we give as input to the strength model for a single move
struct MoveFeatures {
  float winProb;
  float lead;
  float movePolicy;
  float maxPolicy;
  float winrateLoss;  // compared to previous move
  float pointsLoss;  // compared to previous move
};

// The dataset is a chronological sequence of games with move features.
class Dataset {

public:

  // prediction data to be computed by strength model based on recent moves
  struct Prediction {
    float whiteRating;
    float blackRating;
    float score;
  };

  // data on one game from the dataset list file
  struct Game {
    std::string sgfPath;
    struct {
      std::size_t player; // index of player (given as name string in CSV file)
      float rating;       // target provided in input file
      int prevGame;       // index of most recent game with this player before this or -1
      std::vector<MoveFeatures> features; // precomputed from the moves of this player in this game
    } white, black;
    float score;             // game outcome for black: 0 for loss, 1 for win
    Prediction prediction;
    
    enum {
      training = 0,   // is in the training set if ~game.set & 1 is true
      validation = 1, // is in validation set
      batch = 2,      // is in active minibatch
      test = 3        // is in test set
    } set;
  };

  // data on one player
  struct Player {
    std::string name;
    int lastOccurrence; // max index of game where this player participated or -1
  };

  // load the games listed in the path, optionally with move features from featuresDir.
  void load(const std::string& path, const std::string& featureDir = "");
  void store(const std::string& path) const;
  // retrieve up to bufsize moves played by the player in games before the game index, return # retrieved
  size_t getRecentMoves(size_t player, size_t game, MoveFeatures* buffer, size_t bufsize);

  std::vector<Game> games;
  std::vector<Player> players;

  static const uint32_t FEATURE_HEADER; // magic bytes for feature file

private:

  std::map<std::string, std::size_t> nameIndex;  // player names to unique index into player_

  std::size_t getOrInsertNameIndex(const std::string& name);  // insert with lastOccurrence
  void loadFeatures(const std::string& featureDir);
  std::vector<MoveFeatures> readFeaturesFromFile(const std::string& featurePath);

};

// The predictor, given a match between two opponents, estimates their ratings and the match score (win probability).
// This is the abstract base class for our predictors:
//   - The StochasticPredictor based on simple statistics
//   - The SmallPredictor based on the StrengthNet
//   - The FullPredictor (to be done!)
class Predictor {

public:

  // The resulting prediction might keep the players' ratings at 0 (no prediction), but it always predicts the score.
  virtual Dataset::Prediction predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) = 0;

protected:

  // give an expected score by assuming that the given ratings are Elo ratings.
  static float eloScore(float blackRating, float whiteRating);

};

class StochasticPredictor : public Predictor {

public:

  Dataset::Prediction predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) override;

};

#endif  // DATASET_H_
