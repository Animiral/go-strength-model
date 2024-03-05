#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <memory>
#include "sgf.h"
#include "game/board.h"
#include "core/global.h"
#include "core/fileutils.h"

// Safely extract black name and white name from SGFs.
// Preserve only first column (SGF name) and last column (Winner/Score).
// All other columns might be compromised by special characters.

using std::string;
using std::string_view;

namespace {

void printMessage(std::ostream& stream, const string_view& v);
void updateProgressBar(size_t current, size_t total);

}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <source .csv> <target .csv>\n";
        return 1;
    }

    std::ifstream instream(argv[1], std::ifstream::ate);
    auto totalsize = instream.tellg();
    instream.seekg(0);
    std::ofstream outstream(argv[2]);

    string line, sgfpath;
    if(std::getline(instream, line))
        outstream << line << "\n";

    while(std::getline(instream, line)) {
        string sgfpath = Global::trim(line.substr(0, line.find(',')));
        if(sgfpath.empty())
            continue;
        std::unique_ptr<Sgf> sgf(Sgf::loadFile(sgfpath));
        string blackName = sgf->getPlayerNameCompat(P_BLACK);
        string whiteName = sgf->getPlayerNameCompat(P_WHITE);
        outstream << sgfpath << ',' << blackName << ',' << whiteName;
        auto lastComma = line.rfind(',');
        if(string::npos != lastComma)
            outstream << line.substr(lastComma);
        outstream << "\n";

        printMessage(std::cout, "Processed " + sgfpath + "\n");
        updateProgressBar(instream.tellg(), totalsize);
    }

    instream.close();
    outstream.close();

    return 0;
}

namespace {

void printMessage(std::ostream& stream, const string_view& v) {
    stream << "\r\033[J" << v; // messages should end in newline and followed by progress bar update
}

void updateProgressBar(size_t current, size_t total) {
    float progress = (float)current / total;
    int barWidth = 50;

    std::cout << "\r"; // move cursor back to the beginning of the line
    std::cout << "["; // Start of progress bar

    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %";
    std::flush(std::cout); // ensure the progress bar is updated immediately
}

}
