add_library(plane_labeling "plane_labeling.cpp" "include/plane_labeling/plane_labeling.h")
set_property(TARGET plane_labeling PROPERTY CXX_STANDARD 20)

target_link_libraries(plane_labeling PRIVATE plane_detection_utils)

target_include_directories(plane_labeling PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)