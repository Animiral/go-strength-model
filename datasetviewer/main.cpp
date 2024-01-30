#include "global.h"
#include "dataset.h"
#include <vector>
#include <string>
#include <ranges>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cerrno>
#include <cstring>

using Global::strprintf;
using std::cout;
using std::size_t;
using std::vector;
using std::string;
using std::ranges::find;
using std::istringstream;

void help_command(istringstream& args);
void info_command();
void select_command(istringstream& args);
void configure_command(istringstream& args);
void print_command(istringstream& args);
void dump_command(istringstream& args);

enum class Topic { games, moves };
enum class GameColumn { rownum, file, black, white, score, predscore, set };
enum class MoveColumn { rownum, game, color, winprob, lead, policy, maxpolicy, wrloss, ploss };
enum class GameFilter { none, rownum, file, black, white, score, predscore, set };
enum class MoveFilter { none, recent, color };

bool parse_topic(const string& input, Topic& topic);
bool parse_column(const string& input, GameColumn& column);
bool parse_column(const string& input, MoveColumn& column);
bool is_numeric_filter(GameFilter filter);
bool is_numeric_filter(MoveFilter filter);
bool parse_filter(const string& input, GameFilter& filter);
bool parse_filter(const string& input, MoveFilter& filter);
string recentmoves_string(size_t player, size_t game, char delim);
string column_string(size_t rownum, const vector<GameColumn>& entries, char delim);
string column_string(MoveFeatures move, const vector<MoveColumn>& entries, char delim);
vector<bool> selected_games();

struct Selection {
  struct {
    GameFilter filter; // condition on this property
    string pattern; // if string filter: match if it contains the pattern as substring
    float min, max; // if numeric filter: match if it lies between min-max (inclusive)
  } game;
  struct {
    MoveFilter filter; // condition on this property
    string pattern; // if string filter: match if it contains the pattern as substring
    float min, max; // if numeric filter: match if it lies between min-max (inclusive)
  } move;
} selection;

struct Config {
  size_t window = 1000;
} config;

string listpath;
Dataset dataset;

int main(int argc, char* argv[]) {
  if(2 != argc && 3 != argc) {
    istringstream dummy;
    help_command(dummy);
    return EXIT_FAILURE;
  }

  listpath = argv[1];
  string featuredir = argc > 2 ? argv[2] : "";
  dataset.load(listpath, featuredir);
  strprintf(cout, "Dataset Viewer: %d games read from %s (%s features), ready.\n",
    dataset.games.size(), listpath.c_str(), featuredir.empty() ? "no" : "with");

  string line;
  while(cout << "> ", std::getline(std::cin, line)) {
    istringstream stream(line);
    string command;
    stream >> command;
    if(!stream) {
      help_command(stream);
      continue;
    }

    if("info" == command)            info_command();
    else if("select" == command)     select_command(stream);
    else if("configure" == command)  configure_command(stream);
    else if("print" == command)      print_command(stream);
    else if("dump" == command)       dump_command(stream);
    else if("exit" == command)       break;
    else                             help_command(stream);
  }

  cout << "Dataset Viewer: bye!\n";
  return EXIT_SUCCESS;
}

void help_command(istringstream& args) {
  const string select_usage =
    "  select TOPIC FILTER OP VALUE|RANGE    Set the filter for TOPIC.\n"
    "    TOPIC choices: games|moves\n"
    "    FILTER choices for games: none|#|file|black|white|score|predscore|set\n"
    "    FILTER choices for moves: none|recent|color\n"
    "    OP choices: in|contains\n";
  const string configure_usage =
    "  configure SETTING VALUE               Configure a global setting.\n"
    "    SETTING choices: window\n";
  const string print_usage =
    "  print TOPIC COLUMN...                 Write the values to stdout.\n"
    "  dump FILE TOPIC COLUMN...             Write the values to FILE.\n"
    "    TOPIC choices: games|moves\n"
    "    COLUMN choices for games: #|file|black|white|score|predscore|set\n";
    "    COLUMN choices for moves: #|game|color|winprob|lead|policy|maxpolicy|wrloss|ploss\n";
  string section;
  if(args >> section) {
    if("select" == section)
      cout << select_usage <<
        "    ex: select games # in 10-100        (select match records 10-100)\n"
        "    ex: select games black contains tom (select match records where tom plays black)\n"
        "    ex: select moves game in 1          (select moves from games matching game filter)\n"
        "    ex: select moves recent in 10       (select moves in recent set of game 10)\n"
        "    ex: select moves color in black     (select moves in recent set of game 10)\n";
    else if("configure" == section)
      cout << configure_usage <<
        "    ex: configure window 100            (limit recent moves set to 100 moves)\n";
    else if("print" == section || "dump" == section)
      cout << print_usage;
    else goto general_help;
  }
  else {
general_help:
    cout << "Usage: datasetviewer LIST_FILE FEATURE_DIR\n"
      "  View data from the games in the LIST_FILE, with precomputed features stored in FEATURE_DIR.\n"
      "Commands:\n"
      "  help                                  Print this help.\n"
      "  info                                  Print active settings and filters.\n"
      "  exit                                  Exit program.\n"
      << select_usage << configure_usage << print_usage;
  }
}

void info_command() {
  auto print_predicate = [](auto&& selection) {
    if(decltype(selection.filter)::none == selection.filter) {
      cout << "\n";
    }
    else {
      if(is_numeric_filter(selection.filter)) {
        strprintf(cout, " in %f-%f\n", selection.min, selection.max);
      }
      else {
        strprintf(cout, " contains %s\n", selection.pattern.c_str());
      }
    };
  };
  strprintf(cout, "Dataset Viewer: %d games read from %s.\n", dataset.games.size(), listpath.c_str());
  strprintf(cout, "Configuration:\n  window = %d\n", config.window);
  string filterstr = vector<string>({"none", "#", "file", "black", "white", "score", "predscore", "set"})
    [static_cast<size_t>(selection.game.filter)];
  strprintf(cout, "Game Filter: %s", filterstr.c_str());
  print_predicate(selection.game);
  filterstr = vector<string>({"none", "recent", "color"})
    [static_cast<size_t>(selection.move.filter)];
  strprintf(cout, "Move Filter: %s", filterstr.c_str());
  print_predicate(selection.move);
}

template<class Sel> void select_command_condition(istringstream& args, Sel& selection) {
  string filterstr, op, value;
  if(!(args >> filterstr >> op >> value)) {
    cout << "I don't understand your input.\n"
      "Syntax: select TOPIC FILTER OP VALUE|RANGE\n";
    return;
  }
  if("in" != op && "contains" != op) {
    strprintf(cout, "Unknown operator: %s.\n"
      "OP choices: in|contains\n", op.c_str());
    return;
  }

  decltype(selection.filter) filter;
  if(!parse_filter(filterstr, filter))
    return;

  string pattern;
  float min, max;
  if(is_numeric_filter(filter)) {
    char *min_end, *max_end;
    if(size_t index = value.find('-'); string::npos == index) {
      errno = 0;
      min = max = std::strtof(value.c_str(), &min_end);
      if (errno > 0 || min_end != &*value.end())
      {
        strprintf(cout, "I don't understand this value: %s.\n"
          "ex: select games # in 10            (select match record 10)\n", value.c_str());
        return;
      }
    }
    else {
      errno = 0;
      min = std::strtof(value.c_str(), &min_end);
      max = std::strtof(&value[index+1], &max_end);
      if (errno > 0 || min_end != &value[index] || max_end != &*value.end())
      {
        strprintf(cout, "I don't understand this range: %s.\n"
          "ex: select games # in 10-100        (select match records 10-100)\n", value.c_str());
        return;
      }
    }
  }
  else {
    pattern = value;
  }

  selection.filter = filter;
  selection.pattern = pattern;
  selection.min = min;
  selection.max = max;
  cout << "Ok.\n";
}

void select_command(istringstream& args) {
  string topicstr;
  if(!(args >> topicstr)) {
    cout << "I don't understand your input.\n"
      "Syntax: select TOPIC FILTER OP VALUE|RANGE\n";
    return;
  }

  Topic topic;
  if(!parse_topic(topicstr, topic))
    return;

  if(Topic::moves == topic)
    select_command_condition(args, selection.move);
  else
    select_command_condition(args, selection.game);
}

void configure_command(istringstream& args) {
  string setting, valuestr;
  if(!(args >> setting >> valuestr)) {
    cout << "I don't understand your input.\n"
      "Syntax: configure SETTING VALUE\n";
    return;
  }
  if("window" != setting) {
    strprintf(cout, "Unknown setting: %s.\n"
      "SETTING choices: window\n", setting.c_str());
    return;
  }

  errno = 0;
  char* value_end;
  size_t value = std::strtoull(valuestr.c_str(), &value_end, 10);
  if (errno > 0 || value_end != &*valuestr.end())
  {
    strprintf(cout, "I don't understand this value: %s.\n"
      "ex: configure window 100       (limit recent moves set to 100 moves)\n", valuestr.c_str());
    return;
  }

  config.window = value;
  cout << "Ok.\n";
}

template<class Column> vector<Column> print_command_get_columns(istringstream& args) {
  string columnstr;
  if(!(args >> columnstr)) {
    cout << "I don't understand your input.\n"
      "Syntax: print TOPIC COLUMN...\n";
    return {};
  }

  vector<Column> entries;
  do {
    Column column;
    if(!parse_column(columnstr, column))
      continue;
    entries.push_back(column);
  }
  while(args >> columnstr);
  return entries; 
}

void print_command(istringstream& args) {
  string topicstr;
  if(!(args >> topicstr)) {
    cout << "I don't understand your input.\n"
      "Syntax: print TOPIC COLUMN...\n";
    return;
  }

  Topic topic;
  if(!parse_topic(topicstr, topic))
    return;

  vector<bool> selectedGames = selected_games();

  if(Topic::moves == topic) {
    vector<MoveColumn> entries = print_command_get_columns<MoveColumn>(args);
    if(entries.empty())
      return;
    vector<MoveFeatures> blackMoves, whiteMoves;
    for(size_t i = 0; i <= dataset.games.size(); i++) {
      const Dataset::Game& game = dataset.games[i];
      if(!selectedGames[i])
        break;
      if(MoveFilter::recent == selection.move.filter) {
        vector<MoveFeatures> buffer(config.window);
        size_t count = dataset.getRecentMoves(game.black.player, i, buffer.data(), config.window);
        append(blackMoves, buffer, count);
        count = dataset.getRecentMoves(game.white.player, i, buffer.data(), config.window);
        append(whiteMoves, buffer, count);
      }
      else {
        append(blackMoves, game.black.features);
        append(whiteMoves, game.white.features);
      }
    }
    if(MoveFilter::color != selection.move.filter || "black" == selection.move.pattern) {
      for(MoveFeatures move : blackMoves)
        cout << column_string(move, entries, ' ');
    }
    if(MoveFilter::color != selection.move.filter || "white" == selection.move.pattern) {
      for(MoveFeatures move : whiteMoves)
        cout << column_string(move, entries, ' ');
    }
  }
  else {
    vector<GameColumn> entries = print_command_get_columns<GameColumn>(args);
    if(entries.empty())
      return;
    for(size_t i = 0; i < dataset.games.size(); i++) {
      if(selectedGames[i])
        cout << column_string(i, entries, ' ');
    }
  }
}

void dump_command(istringstream& args) {
  string filepath, columnstr;
  if(!(args >> filepath >> columnstr)) {
    cout << "I don't understand your input.\n"
      "Syntax: dump FILE COLUMN...\n";
    return;
  }

  std::ofstream ostrm(filepath);
  if (!ostrm.is_open()) {
    strprintf(cout, "Error: could not write to %s!\n", filepath.c_str());
    return;
  }
  strprintf(cout, "Write to %s...\n", filepath.c_str());

  vector<GameColumn> entries;
  do {
    GameColumn column;
    if(!parse_column(columnstr, column))
      continue;
    entries.push_back(column);
  }
  while(args >> columnstr);

  for(size_t i = 0; i < dataset.games.size(); i++) {
    ostrm << column_string(i, entries, ',');
  }

  ostrm.close();
  cout << "Done.\n";
}

bool parse_topic(const string& input, Topic& topic) {
  vector<string> topics = { "games", "moves" };
  auto topit = find(topics, input);
  if(topics.end() == topit) {
    strprintf(cout, "Unknown topic: %s.\n"
      "TOPIC choices: games|moves\n", input.c_str());
    return false;
  }
  topic = static_cast<Topic>(topit - topics.begin());
  return true;
}

bool parse_column(const string& input, GameColumn& column) {
  vector<string> entries = { "#", "file", "black", "white", "score", "predscore", "set" };
  auto entit = find(entries, input);
  if(entries.end() == entit) {
    strprintf(cout, "Unknown column: %s.\n"
      "COLUMN choices: #|file|black|white|score|predscore|set\n", input.c_str());
    return false;
  }
  column = static_cast<GameColumn>(entit - entries.begin());
  return true;
}

bool parse_column(const string& input, MoveColumn& column) {
  vector<string> entries = { "rownum", "game", "color", "winprob", "lead", "policy", "maxpolicy", "wrloss", "ploss" };
  auto entit = find(entries, input);
  if(entries.end() == entit) {
    strprintf(cout, "Unknown column: %s.\n"
      "COLUMN choices: #|game|color|winprob|lead|policy|maxpolicy|wrloss|ploss\n", input.c_str());
    return false;
  }
  column = static_cast<MoveColumn>(entit - entries.begin());
  return true;
}

bool is_numeric_filter(GameFilter filter) {
  return contains({GameFilter::rownum, GameFilter::score, GameFilter::predscore}, filter);
}

bool is_numeric_filter(MoveFilter filter) {
  return MoveFilter::recent == filter;
}

bool parse_filter(const string& input, GameFilter& filter) {
  vector<string> filters = { "none", "#", "file", "black", "white", "score", "predscore", "set" };
  auto filit = find(filters, input);
  if(filters.end() == filit) {
    strprintf(cout, "Unknown filter: %s.\n"
      "FILTER choices: none|#|file|black|white|score|predscore|set\n", input.c_str());
    return false;
  }
  filter = static_cast<GameFilter>(filit - filters.begin());
  return true;
}

bool parse_filter(const string& input, MoveFilter& filter) {
  vector<string> filters = { "none", "recent", "color" };
  auto filit = find(filters, input);
  if(filters.end() == filit) {
    strprintf(cout, "Unknown filter: %s.\n"
      "FILTER choices: none|recent|color\n", input.c_str());
    return false;
  }
  filter = static_cast<MoveFilter>(filit - filters.begin());
  return true;
}

string column_string(size_t rownum, const vector<GameColumn>& entries, char delim) {
  const Dataset::Game& game = dataset.games[rownum];
  std::ostringstream oss;
  bool first = true;
  for(GameColumn column : entries) {
    if(!first)
      oss << delim;
    first = false;

    switch(column) {
    case GameColumn::rownum: oss << rownum; break;
    case GameColumn::file: oss << game.sgfPath.c_str(); break;
    case GameColumn::black: oss << dataset.players[game.black.player].name.c_str(); break;
    case GameColumn::white: oss << dataset.players[game.white.player].name.c_str(); break;
    case GameColumn::score: oss << game.score; break;
    case GameColumn::predscore: oss << game.prediction.score; break;
    case GameColumn::set: oss << "TVBE"[game.set]; break;
    default: oss << "(unspecified column)";
    }
  }
  oss << "\n";
  return oss.str();
}

string column_string(MoveFeatures move, const vector<MoveColumn>& entries, char delim) {
  std::ostringstream oss;
  bool first = true;
  for(MoveColumn column : entries) {
    if(!first)
      oss << delim;
    first = false;

    switch(column) {
    case MoveColumn::winprob: oss << move.winProb; break;
    case MoveColumn::lead: oss << move.lead; break;
    case MoveColumn::policy: oss << move.movePolicy; break;
    case MoveColumn::maxpolicy: oss << move.maxPolicy; break;
    case MoveColumn::wrloss: oss << move.winrateLoss; break;
    case MoveColumn::ploss: oss << move.pointsLoss; break;
    case MoveColumn::rownum:  // not implemented
    case MoveColumn::game:  // not implemented
    case MoveColumn::color:  // not implemented
    default: oss << "(unspecified column)";
    }
  }
  oss << "\n";
  return oss.str();
}

vector<bool> selected_games() {
  vector<bool> selected(dataset.games.size());
  for(size_t i = 0; i < dataset.games.size(); i++) {
    const Dataset::Game& game = dataset.games[i];

    auto numcheck = [](float n){ return n >= selection.game.min && n <= selection.game.max; };
    auto strcheck = [](const string& s){ return s.contains(selection.game.pattern); };

    bool s = true;
    switch(selection.game.filter) {
    case GameFilter::rownum: if(!numcheck(i)) s = false; break;
    case GameFilter::file: if(!strcheck(game.sgfPath)) s = false; break;
    case GameFilter::black: if(!strcheck(dataset.players[game.black.player].name)) s = false; break;
    case GameFilter::white: if(!strcheck(dataset.players[game.white.player].name)) s = false; break;
    case GameFilter::score: if(!numcheck(game.score)) s = false; break;
    case GameFilter::predscore: if(!numcheck(game.prediction.score)) s = false; break;
    case GameFilter::set: if(!strcheck(string("TVBE"+game.set, 1))) s = false; break;
    case GameFilter::none: default: break;
    }

    switch(selection.move.filter) {
    case MoveFilter::recent: if(i < selection.move.min || i > selection.move.max) s = false; break;
    default: break;
    }

    selected[i] = s;
  }
  return selected;
}
