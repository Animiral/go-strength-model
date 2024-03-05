#include <archive.h>
#include <archive_entry.h>
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <regex>
#include <vector>
#include <memory>
#include "sgf.h"
#include "game/board.h"
#include "core/global.h"
#include "core/fileutils.h"

using std::cout;
using std::string;
using std::string_view;
using Global::strprintf;

const bool DRY_RUN = false;  // just print found games and what would be created/deleted

namespace {

struct CsvLine {
    string path;
    string blackName;
    string whiteName;
    string result;
};

void printMessage(std::ostream& stream, const string_view& v);
void updateProgressBar(size_t current, size_t total);
string makeFilePath(const string& basedir, string date, const string& gameid, const string& blackName, const string& whiteName);
// Return true if the SGF data can be admitted into our dataset, and if so, fill the filePath based on baseDir.
bool isSgfEligible(const string& content, const string& basedir, CsvLine& csvLine);
void extractAndProcess(const std::string& tarPath, const std::string& basedir, const std::string& csvPath);

}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <tar.gz path> <basedir> <.csv path>" << std::endl;
        return 1;
    }

    Board::initHash();

    const std::string tarPath = argv[1];
    const std::string basedir = argv[2];
    const std::string csvPath = argv[3];

    extractAndProcess(tarPath, basedir, csvPath);

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

string makeFilePath(const string& basedir, string date, const string& gameid, const string& blackName, const string& whiteName) {
    Global::replaceAll(date, "-", "/");
    return Global::strprintf("%s/%s/%s-%s-%s.sgf",
        basedir.c_str(), date.c_str(), gameid.c_str(), blackName.c_str(), whiteName.c_str());
}

bool isSgfEligible(const string& content, const string& basedir, CsvLine& csvLine) {
    std::unique_ptr<Sgf> sgf;
    try {
        sgf.reset(Sgf::parse(content));
    }
    catch(const IOError& e) {
        e; // unparseable
        return false;
    }
    if(sgf->nodes.empty())
        return false;
    SgfNode* root = sgf->nodes.front();

    // Date: this becomes the subdirectory path
    string date = root->getSingleProperty("DT");
    // gameid: this becomes the first part of the file name
    string gameid = root->getSinglePropertyOrDefault("PC", "0");
    {
        size_t i = gameid.length() - 1;
        while(i >= 0 && std::isdigit(gameid[i]))
            i--;
        gameid = gameid.substr(i+1);
    }
    try {
        // player names: remainder of the file name
        csvLine.blackName = sgf->getPlayerNameCompat(P_BLACK);
        csvLine.whiteName = sgf->getPlayerNameCompat(P_WHITE);
        csvLine.path = makeFilePath(basedir, date, gameid, csvLine.blackName, csvLine.whiteName);
        csvLine.result = root->getSingleProperty("RE"); // for CSV entry
    }
    catch(const IOError& e) {
        e; // don't even care
        return false;
    }

    // FILTER: only allow 19x19
    XYSize boardsize = sgf->getXYSize();
    try {
        if(19 != boardsize.x || 19 != boardsize.y)
            return false;
    }
    catch(const IOError& e) {
        e; // don't even care
        return false;
    }

    // FILTER: no handicap
    try {
        int handicap = sgf->getHandicapValue();
        if(0 != handicap)
            return false;
    }
    catch(const IOError& e) {
        e; // don't even care
        return false;
    }
    // Other methods of handicap, e.g. 5 black moves at the beginning
    if(root->hasPlacements()) {
        return false;
    }
    std::unique_ptr<CompactSgf> csgf;
    try {
        csgf.reset(new CompactSgf(sgf.get()));
    }
    catch(const IOError& e) {
        e; // we don't handle strange file contents such as AB[ZZ]
        return false;
    }
    int moveCount = csgf->moves.size();
    if(moveCount >= 3 && (csgf->moves[0].pla != P_BLACK || csgf->moves[1].pla != P_WHITE)) {
        return false;
    };

    // FILTER: proper komi
    float komi = sgf->getKomi();
    if(!contains({6.f, 6.5f, 7.f, 7.5f}, komi))
        return false;

    // FILTER: no blitz (<= 5 sec/move)
    //    examples:
    //    TM[5400]OT[25/600 Canadian]  - OTB tournament
    //    TM[259200]OT[86400 fischer]  - correspondence
    //    TM[600]OT[3x30 byo-yomi]     - live
    //    TM[0]OT[259200 simple]       - 3 days per move
    float sec_per_move = sgf->overtimeSecondsPerMove();
    if(sec_per_move <= 5)
        return false;

    // FILTER: short games (<20 moves)
    if(moveCount < 20)
        return false;

    // FILTER: only allow ranked games
    //   examples:
    //   GC[correspondence,unranked]  - comment: unranked game
    string comment = root->getSinglePropertyOrDefault("GC", "");
    if(comment.contains("unranked") || !comment.contains("ranked")) {
        return false;
    }

    // FILTER: only allow results of counting, resignation, timeout
    //   examples:
    //   RE[B+F]   - black wins by forfeit
    //   RE[W+T]   - white wins by time
    static const std::regex result_pattern("[wWbB]\\+[TR\\d]");
    if(!std::regex_search(csvLine.result, result_pattern)) {
        return false;
    }

    // FILTER: all moves in the game must be legal moves
    {
        Rules rules = csgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
        Board board;
        BoardHistory history;
        Player initialPla;
        csgf->setupInitialBoardAndHist(rules, board, initialPla, history);

        for(int turnIdx = 0; turnIdx < csgf->moves.size(); turnIdx++) {
            Move move = csgf->moves[turnIdx];
            if(!history.makeBoardMoveTolerant(board, move.loc, move.pla))
              return false; // illegal move detected
        }
    }

    bool isExists = FileUtils::exists(csvLine.path);
    // File,Player White,Player Black,Winner
    printMessage(std::cout, strprintf("%s: %s vs %s: %s | %s\n",
        csvLine.path.c_str(), csvLine.blackName.c_str(), csvLine.whiteName.c_str(),
        csvLine.result.c_str(), isExists ? "exists" : "notfound"));

    return true;
}

void extractAndProcess(const std::string& tarPath, const std::string& basedir, const std::string& csvPath) {
    struct archive* a;
    struct archive_entry* entry;
    int r;

    a = archive_read_new();
    archive_read_support_format_tar(a);
    archive_read_support_filter_gzip(a);

    if ((r = archive_read_open_filename(a, tarPath.c_str(), 10240)) != ARCHIVE_OK) {
        std::cerr << "Could not open archive: " << tarPath << "\n";
        return;
    }

    std::ofstream csvFile(csvPath, std::ios::out | std::ios::trunc);
    csvFile << "File,Player Black,Player White,Winner\n";

    std::ifstream archivefile(tarPath, std::ifstream::ate | std::ifstream::binary);
    auto totalSize = archivefile.tellg();
    archivefile.close();
    decltype(totalSize) processedSize = 0; // for progress bar

    size_t count = 0; // all read archive entries
    size_t eligibleCount = 0; // files which belong in dataset
    size_t createCount = 0; // files which belong in dataset, but weren't there before
    size_t deleteCount = 0; // files which don't belong in dataset and have been removed
    size_t rejectCount = 0; // files which do not match dataset criteria
    while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
        count++;
        string entryPath = archive_entry_pathname(entry);
        string content;
        const auto fileSize = archive_entry_size(entry);
    // cout << "file size = " << fileSize << "\n";
    // std::cin >> ccc;
        processedSize += fileSize;
        updateProgressBar(processedSize, totalSize);
        content.resize(fileSize);

        r = archive_read_data(a, content.data(), fileSize);
        if (r < 0) {
            printMessage(std::cerr, "Failed to read file content: " + entryPath + "\n");
            continue;
        }
        if (0 == r) { // directories etc
            continue;
        }

        // Process the file content
        CsvLine csvLine;
        bool isEligible = isSgfEligible(content, basedir, csvLine);
        if(isEligible) {
            if(!FileUtils::exists(csvLine.path)) {
                if(DRY_RUN) {
                    printMessage(std::cout, "Would create: " + csvLine.path + "\n");
                }
                else {
                    string dir = FileUtils::dirname(csvLine.path);
                    if(FileUtils::create_directories(dir)) {
                        std::ofstream stream;
                        FileUtils::open(stream, csvLine.path.c_str()); // raises exception when fail
                        stream << content;
                        stream.close();
                    }
                    else {
                        printMessage(std::cerr, "Failed to create dir: " + dir + "\n");
                    }
                }
                createCount++;
            }
            csvFile << Global::strprintf("%s,%s,%s,%s\n",
                csvLine.path.c_str(), csvLine.blackName.c_str(), csvLine.whiteName.c_str(), csvLine.result.c_str());
            eligibleCount++;
        }
        else {
            if(FileUtils::exists(csvLine.path)) {
                if(DRY_RUN) {
                    printMessage(std::cout, "Would delete: " + csvLine.path + "\n");
                }
                else {
                    if(!FileUtils::tryRemoveFile(csvLine.path)) {
                        printMessage(std::cerr, "Failed to delete: " + csvLine.path + "\n");
                    }
                }
                deleteCount++;
            }
            rejectCount++;
        }

        // if(2500 <= eligibleCount)
        //     break;
    }

    archive_read_free(a);
    csvFile.close();

    updateProgressBar(1, 1);
    cout << Global::strprintf("\nDone, found %d eligible SGFs, rejected %d of %d total. %d files created, %d deleted.\n",
        eligibleCount, rejectCount, count, createCount, deleteCount);
}

}
