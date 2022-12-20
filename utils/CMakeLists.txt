file(GLOB SRC "*.cpp")
add_library(utils ${SRC})

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(./common/inc) # for helper_cuda.h

target_include_directories(utils PUBLIC include)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})
