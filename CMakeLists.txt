cmake_minimum_required(VERSION 3.2)

project(neural-network-cpp)

set(CMAKE_PREFIX_PATH /home/srianumakonda/Desktop/libtorch)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main src/main.cpp
		    include/network.h
	      )

target_include_directories(main PRIVATE include)

target_link_libraries(main ${TORCH_LIBRARIES}) 

set_property(TARGET main PROPERTY CXX_STANDARD 14)
