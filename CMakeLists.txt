cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJ_NAME "gpgtpu")
project(${PROJ_NAME} VERSION 1.0)

# Add a series of utils libraries
add_subdirectory(utils)

# Find external packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find GPTPU library
find_library(gptpu /usr/local)

file(GLOB SRC_FILES src/*.cpp)
add_executable(${PROJ_NAME} ${SRC_FILES})
target_include_directories(${PROJ_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJ_NAME} PUBLIC utils ${OpenCV_LIBS} gptpu)

