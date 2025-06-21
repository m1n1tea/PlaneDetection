cmake_minimum_required(VERSION 3.21.0)
project(PlaneDetection VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/opencv4")

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_subdirectory(thirdparty)
add_subdirectory(include)

add_executable(plane_detector plane_detector.cpp)
add_executable(compare_planes compare_planes.cpp)
target_link_libraries(plane_detector PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json plane_detection_utils focal_length_estimation plane_orientation_detection plane_labeling plane_post_processing)
target_link_libraries(compare_planes PRIVATE  ${OpenCV_LIBS} nlohmann_json::nlohmann_json result_evaluation)

set_property(TARGET plane_detector PROPERTY CXX_STANDARD 20)

set_property(TARGET compare_planes PROPERTY CXX_STANDARD 20)




if(${CMAKE_GENERATOR} STREQUAL "Unix Makefiles")
# linux
  install(TARGETS plane_detector)
  install(TARGETS compare_planes)
  install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/generate_predictions_and_output_with_focal_length.sh DESTINATION demo_scripts)
  install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/generate_predictions_and_output.sh DESTINATION demo_scripts)
else()
# windows      
  install(DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/ TYPE BIN)
  install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/generate_predictions_and_output_with_focal_length.ps1 DESTINATION demo_scripts)
  install(PROGRAMS ${CMAKE_SOURCE_DIR}/scripts/generate_predictions_and_output.ps1 DESTINATION demo_scripts)
endif()



install(DIRECTORY ground_truth DESTINATION .)
install(DIRECTORY images DESTINATION .)