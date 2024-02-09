#include "dataset.h"
#include "global.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <memory>

using std::string;
using std::vector;
using std::map;

void Dataset::load(const string& path, const string& featureDir) {
  std::ifstream istrm(path);
  if (!istrm.is_open())
    throw IOError("Could not read dataset from " + path);

  string line;
  std::getline(istrm, line);
  if(!istrm)
    throw IOError("Could not read header line from " + path);
  line = Global::trim(line);

  // clean any previous data
  games.clear();
  players.clear();
  nameIndex.clear();

  // map known fieldnames to row indexes, wherever they may be
  enum class F { ignore, sgfPath, whiteName, blackName, whiteRating, blackRating, score, predictedScore, set };
  vector<F> fields;
  string field;
  std::istringstream iss(line);
  while(std::getline(iss, field, ',')) {
    if("File" == field) fields.push_back(F::sgfPath);
    else if("Player White" == field) fields.push_back(F::whiteName);
    else if("Player Black" == field) fields.push_back(F::blackName);
    else if("WhiteRating" == field) fields.push_back(F::whiteRating);
    else if("BlackRating" == field) fields.push_back(F::blackRating);
    else if("Winner" == field || "Judgement" == field || "Score" == field) fields.push_back(F::score);
    else if("PredictedScore" == field) fields.push_back(F::predictedScore);
    else if("Set" == field) fields.push_back(F::set);
    else fields.push_back(F::ignore);
  }

  while (std::getline(istrm, line)) {
    size_t gameIndex = games.size();
    games.emplace_back();
    Game& game = games[gameIndex];

    line = Global::trim(line);
    iss = std::istringstream(line);
    int fieldIndex = 0;
    while(std::getline(iss, field, ',')) {
      switch(fields[fieldIndex++]) {
      case F::sgfPath:
        game.sgfPath = field;
        break;
      case F::whiteName:
        game.white.player = getOrInsertNameIndex(field);
        break;
      case F::blackName:
        game.black.player = getOrInsertNameIndex(field);
        break;
      case F::whiteRating:
        game.white.rating = Global::stringToFloat(field);
        break;
      case F::blackRating:
        game.black.rating = Global::stringToFloat(field);
        break;
      case F::score:
        if('b' == field[0] || 'B' == field[0])
          game.score = 1;
        else if('w' == field[0] || 'W' == field[0])
          game.score = 0;
        else
          game.score = std::strtof(field.c_str(), nullptr);
        break;
      case F::predictedScore:
        game.prediction.score = std::strtof(field.c_str(), nullptr);
        break;
      case F::set:
        if("t" == field || "T" == field) game.set = Game::training;
        if("v" == field || "V" == field) game.set = Game::validation;
        if("e" == field || "E" == field) game.set = Game::test;
        break;
      default:
      case F::ignore:
        break;
      }
    }
    if(!istrm)
      throw IOError("Error while reading from " + path);
    game.white.prevGame = players[game.white.player].lastOccurrence;
    game.black.prevGame = players[game.black.player].lastOccurrence;

    players[game.white.player].lastOccurrence = gameIndex;
    players[game.black.player].lastOccurrence = gameIndex;
  }

  istrm.close();

  if(!featureDir.empty())
    loadFeatures(featureDir);
}

namespace {
  const char* scoreToString(float score) {
    // only 3 values are really allowed, all perfectly representable in float
    if(0 == score)     return "0";
    if(1 == score)     return "1";
    if(0.5 == score)   return "0.5";
    else               return "(score error)";
  }
}

void Dataset::store(const string& path) const {
  std::ofstream ostrm(path);
  if (!ostrm.is_open())
    throw IOError("Could not write SGF list to " + path);

  ostrm << "File,Player White,Player Black,Score,BlackRating,WhiteRating,PredictedScore,PredictedBlackRating,PredictedWhiteRating,Set\n"; // header

  for(const Game& game : games) {
    string blackName = players[game.black.player].name;
    string whiteName = players[game.white.player].name;

    // file output
    size_t bufsize = game.sgfPath.size() + whiteName.size() + blackName.size() + 100;
    std::unique_ptr<char[]> buffer( new char[ bufsize ] );
    int printed = std::snprintf(buffer.get(), bufsize, "%s,%s,%s,%s,%.2f,%.2f,%.9f,%f,%f,%c\n",
      game.sgfPath.c_str(), whiteName.c_str(), blackName.c_str(),
      scoreToString(game.score), game.black.rating, game.white.rating,
      game.prediction.score, game.prediction.blackRating, game.prediction.whiteRating, "TVBE"[game.set]);
    if(printed <= 0)
      throw IOError("Error during formatting.");
    ostrm << buffer.get();
  }

  ostrm.close();
}

size_t Dataset::getRecentMoves(size_t player, size_t game, MoveFeatures* buffer, size_t bufsize) {
  assert(player < players.size());
  assert(game <= games.size());

  // start from the game preceding the specified index
  int gameIndex;
  if(games.size() == game) {
      gameIndex = players[player].lastOccurrence;
  }
  else {
    Game* gm = &games[game];
    if(player == gm->black.player)
      gameIndex = gm->black.prevGame;
    else if(player == gm->white.player)
      gameIndex = gm->white.prevGame;
    else
      gameIndex = static_cast<int>(game) - 1;
  }

  // go backwards in player's history and fill the buffer in backwards order
  MoveFeatures* outptr = buffer + bufsize;
  while(gameIndex >= 0 && outptr > buffer) {
    while(gameIndex >= 0 && player != games[gameIndex].black.player && player != games[gameIndex].white.player)
      gameIndex--; // this is just defense to ensure that we find a game which the player occurs in
    if(gameIndex < 0)
      break;
    Game* gm = &games[gameIndex];
    bool isBlack = player == gm->black.player;
    const auto& features = isBlack ? gm->black.features : gm->white.features;
    for(int i = features.size(); i > 0 && outptr > buffer;)
      *--outptr = features[--i];
    gameIndex = isBlack ? gm->black.prevGame : gm->white.prevGame;
  }

  // if there are not enough features in history to fill the buffer, adjust
  size_t count = bufsize - (outptr - buffer);
  if(outptr > buffer)
    std::memmove(buffer, outptr, count * sizeof(MoveFeatures));
  return count;
}

const uint32_t Dataset::FEATURE_HEADER = 0xfea70235;

size_t Dataset::getOrInsertNameIndex(const string& name) {
  auto it = nameIndex.find(name);
  if(nameIndex.end() == it) {
    size_t index = players.size();
    players.push_back({name, -1});
    bool success;
    std::tie(it, success) = nameIndex.insert({name, index});
  }
  return it->second;
}

void Dataset::loadFeatures(const string& featureDir) {
  for(Game& game : games) {
    string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
    string blackFeaturesPath = Global::strprintf("%s/%s_BlackFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    string whiteFeaturesPath = Global::strprintf("%s/%s_WhiteFeatures.bin", featureDir.c_str(), sgfPathWithoutExt.c_str());
    game.black.features = readFeaturesFromFile(blackFeaturesPath);
    game.white.features = readFeaturesFromFile(whiteFeaturesPath);
  }
}

vector<MoveFeatures> Dataset::readFeaturesFromFile(const string& featurePath) {
  vector<MoveFeatures> features;
  auto featureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(featurePath.c_str(), "rb"), &std::fclose);
  if(nullptr == featureFile)
    throw IOError("Failed to read access feature file " + featurePath);
  uint32_t header; // must match
  size_t readcount = std::fread(&header, 4, 1, featureFile.get());
  if(1 != readcount || FEATURE_HEADER != header)
    throw IOError("Failed to read from feature file " + featurePath);
  while(!std::feof(featureFile.get())) {
    MoveFeatures mf;
    readcount = std::fread(&mf, sizeof(MoveFeatures), 1, featureFile.get());
    if(1 == readcount)
      features.push_back(mf);
  }
  return features;
}


float Predictor::eloScore(float blackRating, float whiteRating) {
  float Qblack = static_cast<float>(std::pow(10, blackRating / 400));
  float Qwhite = static_cast<float>(std::pow(10, whiteRating / 400));
  return Qblack / (Qblack + Qwhite);
}

// TODO: adapt code from OGS/goratings into glickoScore
    // def expected_win_probability(self, white: "Glicko2Entry", handicap_adjustment: float, ignore_g: bool = False) -> float:
    //     # Implementation extracted from glicko2_update below.
    //     if not ignore_g:
    //         def g() -> float:
    //             return 1
    //     else:
    //         def g() -> float:
    //             return 1 / sqrt(1 + (3 * white.phi ** 2) / (pi ** 2))

    //     E = 1 / (1 + exp(-g() * (self.rating + handicap_adjustment - white.rating) / GLICKO2_SCALE))
    //     return E

namespace {

float fSum(float a[], size_t N) noexcept {  // sum with slightly better numerical stability
  if(N <= 0)
    return 0;
  for(size_t step = 1; step < N; step *= 2) {
    for(size_t i = 0; i+step < N; i += 2*step)
      a[i] += a[i+step];
  }
  return a[0];
}

float fAvg(float a[], size_t N) noexcept {
  return fSum(a, N) / N;
}

float fVar(float a[], size_t N, float avg) noexcept { // corrected variance
  for(size_t i = 0; i < N; i++)
    a[i] = (a[i]-avg)*(a[i]-avg);
  return fSum(a, N) / (N-1);
}

// Because of float limitations, normcdf(x) maxes out for |x| > 5.347.
// Therefore its value is capped such that the result P as well as
// 1.f-P are in the closed interval (0, 1) under float arithmetic.
float normcdf(float x) noexcept {
  float P = .5f * (1.f + std::erf(x / std::sqrt(2.f)));
  if(P >= 1) return std::nextafter(1.f, 0.f);     // =0.99999994f, log(0.99999994f): -5.96e-08
  if(P <= 0) return 1 - std::nextafter(1.f, 0.f); // =0.00000006f, log(0.00000006f): -16.63
  else return P;
}

}

Dataset::Prediction StochasticPredictor::predict(const MoveFeatures* blackFeatures, size_t blackCount, const MoveFeatures* whiteFeatures, size_t whiteCount) {
  if(0 == blackCount || 0 == whiteCount)
    return {0, 0, .5}; // no data for prediction
  constexpr float gamelength = 100; // assume 100 moves per player for an average game
  vector<float> buffer(std::max(blackCount, whiteCount));
  for(size_t i = 0; i < whiteCount; i++)
    buffer[i] = whiteFeatures[i].pointsLoss;
  float wplavg = fAvg(vector<float>(buffer).data(), whiteCount);  // average white points loss
  float wplvar = 2 <= whiteCount ? fVar(buffer.data(), whiteCount, wplavg) : 100.f;      // variance of white points loss
  for(size_t i = 0; i < blackCount; i++)
    buffer[i] = blackFeatures[i].pointsLoss;
  float bplavg = fAvg(vector<float>(buffer).data(), blackCount);  // average black points loss
  float bplvar = 2 <= blackCount ? fVar(buffer.data(), blackCount, bplavg) : 100.f;      // variance of black points loss
  const float epsilon = 0.000001f;  // avoid div by 0
  float z = std::sqrt(gamelength) * (wplavg - bplavg) / std::sqrt(bplvar + wplvar + epsilon); // white pt advantage in standard normal distribution at move# [2*gamelength]
  return {0, 0, normcdf(z)};
}
