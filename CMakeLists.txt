cmake_minimum_required(VERSION 3.18)
project(hyperspectral_disease_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(CUDA REQUIRED)
find_package(GDAL REQUIRED)

set(TENSORRT_ROOT $ENV{TENSORRT_ROOT} CACHE PATH "TensorRT root directory")

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS
        ${TENSORRT_ROOT}/include
        /usr/include
        /usr/local/include
        /usr/local/TensorRT/include
        /opt/tensorrt/include
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib
        /usr/local/TensorRT/lib
        /opt/tensorrt/lib
)

find_library(TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    PATHS
        ${TENSORRT_ROOT}/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib
        /usr/local/TensorRT/lib
        /opt/tensorrt/lib
)

find_library(CUDART_LIBRARY
    NAMES cudart
    PATHS
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib
)

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT include directory not found. Set TENSORRT_ROOT.")
endif()
if(NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT library (nvinfer) not found.")
endif()

message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "GDAL Include: ${GDAL_INCLUDE_DIRS}")
message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY}")

set(SOURCES
    src/main.cpp
    src/envi_reader.cpp
    src/tensorrt_engine.cpp
    src/hyperspectral_engine.cpp
    src/mask_writer.cpp
)

set(HEADERS
    include/envi_reader.h
    include/tensorrt_engine.h
    include/hyperspectral_engine.h
    include/mask_writer.h
)

add_executable(hs_infer ${SOURCES} ${HEADERS})

target_include_directories(hs_infer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${GDAL_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(hs_infer PRIVATE
    ${GDAL_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDART_LIBRARY}
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    target_compile_options(hs_infer PRIVATE -march=armv8.2-a+fp16+simd)
    set(CMAKE_CUDA_ARCHITECTURES "87")
    message(STATUS "Target: Jetson Orin (aarch64, sm_87)")
else()
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
    message(STATUS "Target: x86_64 (sm_75, sm_80, sm_86)")
endif()

set_target_properties(hs_infer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS hs_infer RUNTIME DESTINATION bin)
