file(GLOB SRC "*.cpp")
add_library(utils ${SRC})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_include_directories(utils PUBLIC include)
target_link_libraries(utils PUBLIC ${OpenCV_LIBS})
