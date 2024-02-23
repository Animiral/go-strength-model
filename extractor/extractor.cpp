#include <archive.h>
#include <archive_entry.h>
#include <iostream>
#include <string>
#include <regex>
#include <vector>
#include <memory>
#include "sgf.h"
#include "global.h"
#include "fileutils.h"

using std::cout;
using std::string;
using Global::strprintf;

namespace {

string makeFilePath(const string& basedir, string date, const string& gameid, const string& blackName, const string& whiteName);
// Return true if the SGF data can be admitted into our dataset, and if so, fill the filePath based on baseDir.
bool isSgfEligible(const string& content, const string& basedir, string& filePath);
void extractAndProcess(const std::string& tarPath, const std::string& basedir);

}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <tar.gz path> <basedir>" << std::endl;
        return 1;
    }

    const std::string tarPath = argv[1];
    const std::string basedir = argv[2];

    extractAndProcess(tarPath, basedir);

    return 0;
}

namespace {

string makeFilePath(const string& basedir, string date, const string& gameid, const string& blackName, const string& whiteName) {
    Global::replaceAll(date, "-", "/");
    return Global::strprintf("%s/%s/%s-%s-%s.sgf",
        basedir.c_str(), date.c_str(), gameid.c_str(), blackName.c_str(), whiteName.c_str());
}

bool isSgfEligible(const string& content, const string& basedir, string& filePath) {
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
    // player names: remainder of the file name
    string blackName = sgf->getPlayerName(P_BLACK);
    string whiteName = sgf->getPlayerName(P_WHITE);

    filePath = makeFilePath(basedir, date, gameid, blackName, whiteName);

    // FILTER: only allow 19x19
    try {
        XYSize size = sgf->getXYSize();
        if(19 != size.x || 19 != size.y)
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
    string result = root->getSingleProperty("RE");
    if(!std::regex_search(result, result_pattern)) {
        return false;
    }

    bool isExists = FileUtils::exists(filePath);
    // File,Player White,Player Black,Winner
    std::cout << strprintf("%s: %s vs %s: %s | %s\n",
        filePath.c_str(), blackName.c_str(), whiteName.c_str(), result.c_str(), isExists ? "exists" : "notfound");

    return true;
}

void extractAndProcess(const std::string& tarPath, const std::string& basedir) {
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
        content.resize(fileSize);

        r = archive_read_data(a, content.data(), fileSize);
        if (r < 0) {
            std::cerr << "Failed to read file content: " << entryPath << "\n";
            continue;
        }
        if (0 == r) { // directories etc
            continue;
        }

        // Process the file content
        string filePath;
        bool isEligible = isSgfEligible(content, basedir, filePath);
        if(isEligible) {
            if(!FileUtils::exists(filePath)) {
                // std::cout << "Would create: " << filePath << "\n";
                // string dir = FileUtils::dirname(filePath);
                // if(FileUtils::create_directories(dir)) {
                //     std::ofstream stream(filePath);
                //     stream << content;
                //     stream.close();
                // }
                // else {
                //     std::cerr << "Failed to create dir: " << dir << "\n";
                // }
                createCount++;
            }
            eligibleCount++;
        }
        else {
            if(FileUtils::exists(filePath)) {
                // std::cout << "Would delete: " << filePath << "\n";
                // if(!FileUtils::tryRemoveFile(filePath)) {
                //     std::cerr << "Failed to delete: " << filePath << "\n";
                // }
                deleteCount++;
            }
            rejectCount++;
        }

        // if(2500 <= eligibleCount)
        //     break;
    }

    archive_read_free(a);

    std::cout << Global::strprintf("Done, found %d eligible SGFs, rejected %d of %d total. %d files created, %d deleted.\n",
        eligibleCount, rejectCount, count, createCount, deleteCount);
}

}
