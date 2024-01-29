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

using std::cout;
using std::size_t;
using std::vector;
using std::string;
using std::ranges::find;
using std::istringstream;

std::ostream& strprintf(std::ostream& stream, const char* fmt, std::va_list ap);
std::ostream& strprintf(std::ostream& stream, const char* fmt, ...);
template<class T> bool contains(std::initializer_list<T> list, const T& value);
void help_command();
void info_command();
void select_command(istringstream& args);
void configure_command(istringstream& args);
void print_command(istringstream& args);
void dump_command(istringstream& args);

enum class Entry { all, rownum, file, black, white, score, predscore, set, recentmoves };
enum class Property { none, rownum, file, black, white, score, predscore, set };

bool parse_entry(const string& input, Entry& entry);
bool is_numeric_property(Property property);
bool parse_property(const string& input, Property& property);

struct Selection {
  Entry entry; // what to extract
  struct {
    Property property; // condition on this property
    string pattern; // if string property: match if it contains the pattern as substring
    float min, max; // if numeric property: match if it lies between min-max (inclusive)
  } filter;
} selection;

struct {
  size_t window = 1000;
} config;

string listpath;

int main(int argc, char* argv[]) {
  if(3 != argc) {
    help_command();
    return EXIT_FAILURE;
  }

  listpath = argv[1];
  string featuredir = argv[2];
  strprintf(cout, "Dataset Viewer: **** games read from %s, ready.\n", listpath.c_str());

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

std::ostream& strprintf(std::ostream& stream, const char* fmt, std::va_list ap) {
  std::va_list ap_copy;
  va_copy(ap_copy, ap);
  int size = vsnprintf(nullptr, 0, fmt, ap_copy);
  va_end(ap_copy);

  if (size < 0) {
    std::cerr << "Error: " << std::strerror(errno) << std::endl;
    std::abort();
  }

  string buffer(size + 1, '\0');
  vsnprintf(buffer.data(), buffer.size(), fmt, ap);
  return stream << buffer;
}

std::ostream& strprintf(std::ostream& stream, const char* fmt, ...) {
  std::va_list ap;
  va_start(ap, fmt);
  std::ostream& retstream = strprintf(stream, fmt, ap);
  va_end(ap);
  return retstream;
}

template<class T>
bool contains(std::initializer_list<T> list, const T& value) {
  for(auto it = list.begin(), end = list.end(); it != end; ++it) {
    if(value == *it)
      return true;
  }
  return false;
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
    "    ENTRY choices: all|#|file|black|white|score|predscore|set|recentmoves\n";
}

void info_command() {
  strprintf(cout, "Dataset Viewer: **** games read from %s.\n", listpath.c_str());
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

  do {
    Entry entry;
    if(!parse_entry(entrystr, entry))
      continue;

    // TODO!
    cout << "PRINT entry: " << entrystr << "\n";
  }
  while(args >> entrystr);
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

  do {
    Entry entry;
    if(!parse_entry(entrystr, entry))
      continue;

    // TODO!
    cout << "DUMP entry: " << entrystr << "\n";
  }
  while(args >> entrystr);

  ostrm.close();
  cout << "Done.\n";
}

bool parse_entry(const string& input, Entry& entry) {
  vector<string> entries = { "all", "#", "file", "black", "white", "score", "predscore", "set", "recentmoves" };
  auto entit = find(entries, input);
  if(entries.end() == entit) {
    strprintf(cout, "Unknown entry: %s.\n"
      "ENTRY choices: all|#|file|black|white|score|predscore|set|recentmoves\n", input.c_str());
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
