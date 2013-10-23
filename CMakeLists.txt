cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(mypcl)

find_package(PCL 1.3 REQUIRED COMPONENTS)
find_package(OpenCV REQUIRED)
find_package(Boost 1.40 COMPONENTS program_options REQUIRED )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(mypcl main.cpp mysvm.cpp)
target_link_libraries(mypcl ${PCL_LIBRARIES})
target_link_libraries(mypcl ${OpenCV_LIBS})
target_link_libraries(mypcl ${Boost_LIBRARIES} )

add_executable(dbgtool dbgtool.cpp mysvm.cpp)
target_link_libraries(dbgtool ${PCL_LIBRARIES})
target_link_libraries(dbgtool ${OpenCV_LIBS})
target_link_libraries(dbgtool ${Boost_LIBRARIES} )

add_definitions(-std=c++11)
