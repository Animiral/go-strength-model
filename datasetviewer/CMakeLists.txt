cmake_minimum_required(VERSION 3.18.2) # Required for CUDA language support
project(datasetviewer)

# Set C++ standard and ensure compiler is g++
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++)

# Enable CUDA support
enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
#add_executable(${PROJECT_NAME} main.cpp kernel.cu)

# CUDA stuff
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 11)
#set(CUDA_STANDARD 11) # from KataGo

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
