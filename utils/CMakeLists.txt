file(GLOB SRC "*.cpp")
add_library(utils ${SRC})
target_include_directories(utils PUBLIC include)
