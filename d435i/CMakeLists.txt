add_executable(d435i-liveness central_depth_liveness.cpp)
target_link_libraries(d435i-liveness PRIVATE realsense2)
target_compile_features(d435i-liveness PRIVATE cxx_std_17)
