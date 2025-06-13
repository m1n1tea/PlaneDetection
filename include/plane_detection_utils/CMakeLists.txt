add_library(plane_detection_utils "plane_detection_utils.cpp" "include/plane_detection_utils/plane_detection_utils.h")
set_property(TARGET plane_detection_utils PROPERTY CXX_STANDARD 20)

target_link_libraries(plane_detection_utils PUBLIC ${OpenCV_LIBS})

target_include_directories(plane_detection_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)