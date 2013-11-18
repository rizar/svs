cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(svs)

set(CMAKE_CXX_COMPILER g++-4.8)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
include(cotire)

find_package(PCL 1.3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED )
find_package(OpenMP)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_definitions(-std=c++11 -fopenmp -Wall -Wextra -Wno-sign-compare)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory("${PROJECT_SOURCE_DIR}/components")
add_subdirectory("${PROJECT_SOURCE_DIR}/tools")

