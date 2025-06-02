cmake_minimum_required(VERSION 3.18.0)
project(PlaneDetection VERSION 0.1.0 LANGUAGES CXX)

set(OpenCV_ROOT "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv4")

find_package(OpenCV REQUIRED)
find_package(casadi CONFIG REQUIRED)
add_subdirectory(thirdparty)

add_library(semcv ./semcv.cpp ./semcv.h)
set_property(TARGET semcv PROPERTY CXX_STANDARD 20)
target_link_libraries(semcv PRIVATE ${OpenCV_LIBS} casadi::casadi JLinkage)

target_include_directories(semcv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
  $<INSTALL_INTERFACE:.>
)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE ${OpenCV_LIBS} semcv casadi::casadi)

set_property(TARGET main PROPERTY CXX_STANDARD 20)
