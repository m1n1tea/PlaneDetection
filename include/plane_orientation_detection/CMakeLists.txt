add_library(plane_orientation_detection "plane_orientation_detection.cpp" "include/plane_orientation_detection/plane_orientation_detection.h")
set_property(TARGET plane_orientation_detection PROPERTY CXX_STANDARD 20)

target_link_libraries(plane_orientation_detection PRIVATE plane_detection_utils)

target_include_directories(plane_orientation_detection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)