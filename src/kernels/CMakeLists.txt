file(GLOB SRC "*.cpp" "*.cu")
add_library(kernels ${SRC})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
  
target_include_directories(kernels PUBLIC include ../../include ../../utils/include)
target_link_libraries(kernels PUBLIC ${OpenCV_LIBS})
