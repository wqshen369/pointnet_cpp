cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project (pointnet_cls)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -W")
# set(PCL_DIR /usr/local/share/pcl-1.8/PCLConfig.cmake)
find_package(PCL 1.8 REQUIRED)

# set(PCL_LIBRARY_DIRS /usr/lib)

include_directories(
    /usr/local/tensorflow/include
    ${PCL_INCLUDE_DIRS}
    ) 
link_directories(
    /usr/local/tensorflow/lib
    ${PCL_LIBRARY_DIRS}
    )
add_definitions(${PCL_DEFINITIONS})

add_executable(pointnet_cls  pointnet_cls.cpp) 
target_link_libraries(pointnet_cls tensorflow_cc tensorflow_framework ${PCL_LIBRARIES})
