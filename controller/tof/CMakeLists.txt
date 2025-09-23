cmake_minimum_required(VERSION 3.10)
project(tof_reader)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vl53l0x STATIC
    third_party/vl53l0x/src/vl53l0x.cpp
    third_party/vl53l0x/src/interfaces/i2cdev.cpp
    third_party/vl53l0x/src/interfaces/i2c_interface.cpp
)

target_include_directories(vl53l0x PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vl53l0x/include
)

target_link_libraries(vl53l0x PUBLIC i2c)

add_executable(tof-reader
    src/main.cpp
    src/tof_reader.cpp
)

target_include_directories(tof-reader PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/vl53l0x/include
)

target_link_libraries(tof-reader PRIVATE vl53l0x)

target_link_libraries(tof-reader PRIVATE i2c)
