# Set SOURCES variable
set(SOURCES
    src/JLinkage.cpp
    src/RandomSampler.cpp
    src/Utilities.cpp
    src/VPCluster.cpp
    src/VPSample.cpp
    src/updator.cpp
)

# Set HEADERS_PUBLIC variable
set(HEADERS_PUBLIC
    include/JLinkage.h
    include/RandomSampler.h
    include/updator.h
    include/VPCluster.h
    include/VPSample.h
)

set(LIBRARY_NAME  "JLinkage")

# Set HEADERS_PRIVATE variable
file(GLOB BM_HEADERS "include/bm/*")
file(GLOB KDTREE_HEADERS "include/kdtree++/*")

set(HEADERS_PRIVATE
    include/PrimitiveFunctions.h
    include/Utilities.h
    include/VPPrimitive.h
    ${BM_HEADERS}
    ${KDTREE_HEADERS}
)

# library configuration
add_library(${LIBRARY_NAME}
  ${SOURCES}
  ${HEADERS_PUBLIC}
  ${HEADERS_PRIVATE}
  )

target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)

target_include_directories(
  ${LIBRARY_NAME}
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:./include>"
)

target_include_directories(${LIBRARY_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/JLinkage/include)


