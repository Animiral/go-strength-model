cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(extractor VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the libarchive package
find_package(LibArchive REQUIRED)

# Add the executable
add_executable(extractor extractor.cpp sgf.cpp fileutils.cpp global.cpp rules.cpp board.cpp)

# Link the executable to the libarchive library
target_link_libraries(extractor ${LibArchive_LIBRARIES})

# Include directories for libarchive
target_include_directories(extractor PRIVATE ${LibArchive_INCLUDE_DIRS})
