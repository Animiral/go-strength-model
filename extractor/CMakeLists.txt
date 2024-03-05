cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(extractor VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the libarchive package
find_package(LibArchive REQUIRED)

# Add the executable
add_executable(extractor extractor.cpp
	sgf.cpp
	core/base64.cpp         core/global.cpp       core/rand_helpers.cpp
	core/bsearch.cpp        core/hash.cpp         core/sha2.cpp
	core/commandloop.cpp    core/logger.cpp       core/test.cpp
	core/config_parser.cpp  core/mainargs.cpp     core/threadsafecounter.cpp
	core/datetime.cpp       core/threadsafequeue.cpp
	core/elo.cpp            core/md5.cpp          core/threadtest.cpp
	core/fancymath.cpp      core/multithread.cpp  core/timer.cpp
	core/fileutils.cpp      core/rand.cpp
	game/board.cpp  game/boardhistory.cpp  game/graphhash.cpp  game/rules.cpp)

# Link the executable to the libarchive library
target_link_libraries(extractor ${LibArchive_LIBRARIES})

# Include directories for libarchive
target_include_directories(extractor PRIVATE ${LibArchive_INCLUDE_DIRS})
