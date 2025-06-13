add_library(focal_length_estimation "focal_length_estimation.cpp" "include/focal_length_estimation/focal_length_estimation.h")
set_property(TARGET focal_length_estimation PROPERTY CXX_STANDARD 20)

target_link_libraries(focal_length_estimation PRIVATE JLinkage plane_detection_utils)

target_include_directories(focal_length_estimation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)