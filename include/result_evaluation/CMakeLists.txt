add_library(result_evaluation "result_evaluation.cpp" "include/result_evaluation/result_evaluation.h")
set_property(TARGET result_evaluation PROPERTY CXX_STANDARD 20)

target_link_libraries(result_evaluation PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

target_include_directories(result_evaluation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)