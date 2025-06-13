add_library(plane_post_processing "plane_post_processing.cpp" "include/plane_post_processing/plane_post_processing.h")
set_property(TARGET plane_post_processing PROPERTY CXX_STANDARD 20)

target_link_libraries(plane_post_processing PRIVATE ${OpenCV_LIBS})

target_include_directories(plane_post_processing PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)