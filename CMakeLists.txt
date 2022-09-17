cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(gpgtpu VERSION 1.0)

# Add a series of utils libraries
add_subdirectory(utils)

# Find external packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find GPTPU library
find_library(gptpu /usr/local)

add_executable(gpgtpu ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
target_link_libraries(gpgtpu PUBLIC utils ${OpenCV_LIBS} gptpu)

