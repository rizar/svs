cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(svs)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")
include(cotire)

find_package(PCL 1.3 REQUIRED COMPONENTS)
find_package(OpenCV REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions(-std=c++11)
add_definitions(-fopenmp)

add_executable(view main.cpp mysvm.cpp)
target_link_libraries(view ${PCL_LIBRARIES})
target_link_libraries(view ${OpenCV_LIBS})
target_link_libraries(view ${Boost_LIBRARIES} )
cotire(view)

add_executable(dbgtool dbgtool.cpp mysvm.cpp)
target_link_libraries(dbgtool ${PCL_LIBRARIES})
target_link_libraries(dbgtool ${OpenCV_LIBS})
target_link_libraries(dbgtool ${Boost_LIBRARIES} )
cotire(dbgtool)

add_executable(fpoints featurepoints.cpp searcher.cpp mysvm.cpp)
target_link_libraries(fpoints ${PCL_LIBRARIES})
target_link_libraries(fpoints ${OpenCV_LIBS})
target_link_libraries(fpoints ${Boost_LIBRARIES} )
cotire(fpoints)

