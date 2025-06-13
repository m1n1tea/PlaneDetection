cmake_minimum_required(VERSION 3.18.0)
project(PlaneDetection VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/opencv4")

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

add_subdirectory(thirdparty)
add_subdirectory(include)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json plane_detection_utils focal_length_estimation plane_orientation_detection plane_labeling plane_post_processing)

set_property(TARGET main PROPERTY CXX_STANDARD 20)
