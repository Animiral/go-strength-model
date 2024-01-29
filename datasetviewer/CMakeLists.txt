cmake_minimum_required(VERSION 3.18.2) # Required for CUDA language support
project(datasetviewer)

# Set C++ standard and ensure compiler is g++
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++)

add_executable(${PROJECT_NAME} main.cpp global.cpp dataset.cpp)
#add_executable(${PROJECT_NAME} main.cpp kernel.cu)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
