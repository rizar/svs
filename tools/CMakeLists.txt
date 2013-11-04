add_executable(runsvs runsvs.cpp)
target_link_libraries(runsvs components ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable(runiss runiss.cpp)
target_link_libraries(runiss components ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(dbgtool dbgtool.cpp)
target_link_libraries(dbgtool components ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(evaltool evaltool.cpp)
target_link_libraries(evaltool components ${PCL_LIBRARIES} ${Boost_LIBRARIES})

add_executable(testnewsvm testnewsvm.cpp)
target_link_libraries(testnewsvm components ${PCL_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})

