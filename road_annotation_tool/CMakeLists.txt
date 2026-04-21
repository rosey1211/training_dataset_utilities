cmake_minimum_required(VERSION 3.14)
project(image_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(image_viewer image_viewer.cpp)
target_include_directories(image_viewer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_viewer PRIVATE ${OpenCV_LIBS})
