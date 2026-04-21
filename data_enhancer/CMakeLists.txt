cmake_minimum_required(VERSION 3.14)
project(augment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(augment augment_v2.cpp)
target_include_directories(augment PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(augment PRIVATE ${OpenCV_LIBS})
