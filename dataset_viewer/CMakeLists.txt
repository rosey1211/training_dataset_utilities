cmake_minimum_required(VERSION 3.10)
project(dataset_viewer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(dataset_viewer dataset_viewer.cpp)
target_link_libraries(dataset_viewer ${OpenCV_LIBS})
