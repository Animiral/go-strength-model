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

void help_command();
void info_command();
void select_command(istringstream& args);
void configure_command(istringstream& args);
void print_command(istringstream& args);
void dump_command(istringstream& args);

enum class Entry { rownum, file, black, white, score, predscore, set, black_recentmoves, white_recentmoves };
enum class Property { none, rownum, file, black, white, score, predscore, set };

bool parse_entry(const string& input, Entry& entry);
bool is_numeric_property(Property property);
bool parse_property(const string& input, Property& property);
string recentmoves_string(size_t player, size_t game, char delim);
string selection_string(size_t rownum, const vector<Entry>& entries, char delim);

struct Selection {
  struct {
    Property property; // condition on this property
    string pattern; // if string property: match if it contains the pattern as substring
    float min, max; // if numeric property: match if it lies between min-max (inclusive)
  } filter;
} selection;

struct Config {
  size_t window = 1000;
} config;

string listpath;
Dataset dataset;

int main(int argc, char* argv[]) {
  if(2 != argc && 3 != argc) {
    help_command();
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
      help_command();
      continue;
    }

    if("info" == command)            info_command();
    else if("select" == command)     select_command(stream);
    else if("configure" == command)  configure_command(stream);
    else if("print" == command)      print_command(stream);
    else if("dump" == command)       dump_command(stream);
    else if("exit" == command)       break;
    else                             help_command();
  }

  cout << "Dataset Viewer: bye!\n";
  return EXIT_SUCCESS;
}

void help_command() {
  cout << "Usage: datasetviewer LIST_FILE FEATURE_DIR\n"
    "  View data from the games in the LIST_FILE, with precomputed features stored in FEATURE_DIR.\n"
    "Commands:\n"
    "  help                               Print this help.\n"
    "  info                               Print active settings and filters.\n"
    "  exit                               Exit program.\n"
    "  select PROPERTY OP VALUE|RANGE     Set the filter.\n"
    "    PROPERTY choices: none|#|file|black|white|score|predscore|set\n"
    "    OP choices: in|contains\n"
    "    ex: select # in 10-100         (select match records 10-100)\n"
    "    ex: select black contains tom  (select match records where tom plays black)\n"
    "  configure SETTING VALUE            Configure a global setting.\n"
    "    SETTING choices: window\n"
    "    ex: configure window 100       (limit recent moves set to 100 moves)\n"
    "  print ENTRY...                     Write the values to stdout.\n"
    "  dump FILE ENTRY...                 Write the values to FILE.\n"
    "    ENTRY choices: #|file|black|white|score|predscore|set|black_recentmoves|white_recentmoves\n";
}

void info_command() {
  strprintf(cout, "Dataset Viewer: %d games read from %s.\n", dataset.games.size(), listpath.c_str());
  strprintf(cout, "Configuration:\n  window = %d\n", config.window);
  string propertystr = vector<string>({"none", "#", "file", "black", "white", "score", "predscore", "set"})
    [static_cast<size_t>(selection.filter.property)];
  strprintf(cout, "Filter: %s", propertystr.c_str());
  if(Property::none == selection.filter.property) {
    cout << "\n";
  }
  else {
    if(is_numeric_property(selection.filter.property)) {
      strprintf(cout, " in %f-%f\n", selection.filter.min, selection.filter.max);
    }
    else {
      strprintf(cout, " contains %s\n", selection.filter.pattern.c_str());
    }
  }
}

void select_command(istringstream& args) {
  string propertystr, op, value;
  if(!(args >> propertystr >> op >> value)) {
    cout << "I don't understand your input.\n"
      "Syntax: select PROPERTY OP VALUE|RANGE\n";
    return;
  }
  if("in" != op && "contains" != op) {
    strprintf(cout, "Unknown operator: %s.\n"
      "OP choices: in|contains\n", op.c_str());
    return;
  }

  Property property;
  if(!parse_property(propertystr, property))
    return;

  string pattern;
  float min, max;
  if(is_numeric_property(property)) {
    char *min_end, *max_end;
    if(size_t index = value.find('-'); string::npos == index) {
      errno = 0;
      min = max = std::strtof(value.c_str(), &min_end);
      if (errno > 0 || min_end != &*value.end())
      {
        strprintf(cout, "I don't understand this value: %s.\n"
          "ex: select # in 10             (select match record 10)\n", value.c_str());
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
          "ex: select # in 10-100         (select match records 10-100)\n", value.c_str());
        return;
      }
    }
  }
  else {
    pattern = value;
  }

  selection.filter.property = property;
  selection.filter.pattern = pattern;
  selection.filter.min = min;
  selection.filter.max = max;
  cout << "Ok.\n";
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

void print_command(istringstream& args) {
  string entrystr;
  if(!(args >> entrystr)) {
    cout << "I don't understand your input.\n"
      "Syntax: print ENTRY...\n";
    return;
  }

  vector<Entry> entries;
  do {
    Entry entry;
    if(!parse_entry(entrystr, entry))
      continue;
    entries.push_back(entry);
  }
  while(args >> entrystr);

  for(size_t i = 0; i < dataset.games.size(); i++) {
    cout << selection_string(i, entries, ' ');
  }
}

void dump_command(istringstream& args) {
  string filepath, entrystr;
  if(!(args >> filepath >> entrystr)) {
    cout << "I don't understand your input.\n"
      "Syntax: dump FILE ENTRY...\n";
    return;
  }

  std::ofstream ostrm(filepath);
  if (!ostrm.is_open()) {
    strprintf(cout, "Error: could not write to %s!\n", filepath.c_str());
    return;
  }
  strprintf(cout, "Write to %s...\n", filepath.c_str());

  vector<Entry> entries;
  do {
    Entry entry;
    if(!parse_entry(entrystr, entry))
      continue;
    entries.push_back(entry);
  }
  while(args >> entrystr);

  for(size_t i = 0; i < dataset.games.size(); i++) {
    ostrm << selection_string(i, entries, ',');
  }

  ostrm.close();
  cout << "Done.\n";
}

bool parse_entry(const string& input, Entry& entry) {
  vector<string> entries = { "#", "file", "black", "white", "score", "predscore", "set", "black_recentmoves", "white_recentmoves" };
  auto entit = find(entries, input);
  if(entries.end() == entit) {
    strprintf(cout, "Unknown entry: %s.\n"
      "ENTRY choices: #|file|black|white|score|predscore|set|black_recentmoves|white_recentmoves\n", input.c_str());
    return false;
  }
  entry = static_cast<Entry>(entit - entries.begin());
  return true;
}

bool is_numeric_property(Property property) {
  return contains({Property::rownum, Property::score, Property::predscore}, property);
}

bool parse_property(const string& input, Property& property) {
  vector<string> properties = { "none", "#", "file", "black", "white", "score", "predscore", "set" };
  auto propit = find(properties, input);
  if(properties.end() == propit) {
    strprintf(cout, "Unknown property: %s.\n"
      "PROPERTY choices: none|#|file|black|white|score|predscore|set\n", input.c_str());
    return false;
  }
  property = static_cast<Property>(propit - properties.begin());
  return true;
}

string recentmoves_string(size_t player, size_t game, char delim) {
  vector<MoveFeatures> buffer(config.window);
  buffer.resize(dataset.getRecentMoves(player, game, buffer.data(), config.window));
  std::ostringstream oss;
  for(MoveFeatures& mf : buffer) {
    strprintf(oss, "%f%c%f%c%f%c%f%c%f%c%f\n",
      mf.winProb, delim, mf.lead, delim, mf.movePolicy, delim,
      mf.maxPolicy, delim, mf.winrateLoss, delim, mf.pointsLoss);
  }
  return oss.str();
}

string selection_string(size_t rownum, const vector<Entry>& entries, char delim) {
  const Dataset::Game& game = dataset.games[rownum];

  auto numcheck = [](float n){ return n >= selection.filter.min && n <= selection.filter.max; };
  auto strcheck = [](const string& s){ return s.contains(selection.filter.pattern); };

  switch(selection.filter.property) {
  case Property::none: default: break;
  case Property::rownum: if(numcheck(rownum)) break; else return "";
  case Property::file: if(strcheck(game.sgfPath)) break; else return "";
  case Property::black: if(strcheck(dataset.players[game.black.player].name)) break; else return "";
  case Property::white: if(strcheck(dataset.players[game.white.player].name)) break; else return "";
  case Property::score: if(numcheck(game.score)) break; else return "";
  case Property::predscore: if(numcheck(game.prediction.score)) break; else return "";
  case Property::set: if(strcheck(string("TVBE"+game.set, 1))) break; else return "";
  }

  std::ostringstream oss;
  bool first = true;
  for(Entry entry : entries) {
    if(!first)
      oss << delim;
    first = false;

    switch(entry) {
    case Entry::rownum: oss << rownum; break;
    case Entry::file: oss << game.sgfPath.c_str(); break;
    case Entry::black: oss << dataset.players[game.black.player].name.c_str(); break;
    case Entry::white: oss << dataset.players[game.white.player].name.c_str(); break;
    case Entry::score: oss << game.score; break;
    case Entry::predscore: oss << game.prediction.score; break;
    case Entry::set: oss << "TVBE"[game.set]; break;
    case Entry::black_recentmoves: oss << recentmoves_string(game.black.player, rownum, delim); break;
    case Entry::white_recentmoves: oss << recentmoves_string(game.white.player, rownum, delim); break;
    default: oss << "(unspecified entry)";
    }
  }
  oss << "\n";
  return oss.str();
}
