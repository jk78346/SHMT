file(GLOB SRC "*.cpp" "*.cu")
add_library(kernels ${SRC})

list(APPEND CUDA_DEV_LIBRARIES
            ${CUDA_cufft_LIBRARY})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
  
target_include_directories(kernels PUBLIC include ../../include ../../utils/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(kernels PUBLIC utils ${OpenCV_LIBS} ${CUDA_DEV_LIBRARIES})
