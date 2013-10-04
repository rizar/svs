cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(mypcl)

find_package(PCL 1.3 REQUIRED COMPONENTS)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(mypcl main.cpp)
target_link_libraries(mypcl ${PCL_LIBRARIES})
target_link_libraries(mypcl ${OpenCV_LIBS})
