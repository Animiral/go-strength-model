// Filter games from a dataset based on criteria.
// If the games in the input CSV do not meet the criteria, they are dropped from the output.

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include "sgf.h"
#include "game/board.h"
#include "core/global.h"
#include "core/fileutils.h"

using std::string;
using std::vector;

namespace {

int findColumnIndex(const vector<string>& headers, const string& columnName);
vector<string> splitRow(const string& row);
bool isSgfEligible(const string& filename);

}

// Main function to filter CSV rows
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.csv path> <output.csv path>" << std::endl;
        return 1;
    }

    Board::initHash();

    const string inputpath = argv[1];
    const string outputpath = argv[2];

    std::ifstream inputFile(inputpath);
    std::ofstream outputFile(outputpath);
    string row;

    if (!inputFile.is_open() || !outputFile.is_open()) {
        std::cerr << "Error opening files!" << std::endl;
        return 1;
    }

    // header row is required
    std::getline(inputFile, row);
    int fileCol = findColumnIndex(splitRow(row), "File");
    if (-1 == fileCol) {
        std::cerr << "\"File\" column not found." << std::endl;
        return 1;
    }

    outputFile << row << std::endl;

    int good = 0, bad = 0;

    while (std::getline(inputFile, row)) {
        auto columns = splitRow(row);
        if (isSgfEligible(columns[fileCol])) {
            outputFile << row << std::endl;
            good++;
        }
        else {
        	bad++;
        }
    	std::cout << "\rRows: " << good << " OK, " << bad << " bad..."; // move cursor back to the beginning of the line
    }

    std::cout << "\rRows: " << good << " OK, " << bad << " bad, " << (good+bad) << " total.\n";

    inputFile.close();
    outputFile.close();

    return 0;
}

namespace {

int findColumnIndex(const vector<string>& headers, const string& columnName) {
    for (size_t i = 0; i < headers.size(); ++i) {
        if (headers[i] == columnName) {
            return i;
        }
    }
    return -1;
}

vector<string> splitRow(const string& row) {
    vector<string> columns;
    std::stringstream rowStream(row);
    string column;
    
    while (std::getline(rowStream, column, ',')) {
        columns.push_back(column);
    }

    return columns;
}

bool isSgfEligible(const string& filename) {
    std::unique_ptr<Sgf> sgf;
    try {
        sgf.reset(Sgf::loadFile(filename));
    }
    catch(const IOError& e) {
        e; // unparseable
        return false;
    }
    if(sgf->nodes.empty())
        return false;
    SgfNode* root = sgf->nodes.front();

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
    if(!sgf->nodes[0]->hasProperty("KM"))
        return false; // override sgf->getKomi() default because we do require Komi specification
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
    try {
	    string result = root->getSingleProperty("RE");
    	static const std::regex result_pattern("[wWbB]\\+[TR\\d]");
	    if(!std::regex_search(result, result_pattern)) {
	        return false;
	    }
    }
    catch(const IOError& e) {
        e; // don't even care
        return false;
    }

    // FILTER: all moves in the game must be legal moves, no passes up to move 50
    {
        Rules rules = csgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
        Board board;
        BoardHistory history;
        Player initialPla;
        csgf->setupInitialBoardAndHist(rules, board, initialPla, history);

        for(int turnIdx = 0; turnIdx < csgf->moves.size(); turnIdx++) {
            Move move = csgf->moves[turnIdx];
            if(turnIdx < 50 && (Board::PASS_LOC == move.loc || Board::NULL_LOC == move.loc))
                return false;
            if(!history.makeBoardMoveTolerant(board, move.loc, move.pla))
              return false; // illegal move detected
        }
    }

    return true;
}

}
