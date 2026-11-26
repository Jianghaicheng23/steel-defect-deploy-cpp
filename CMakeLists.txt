cmake_minimum_required(VERSION 3.16)
project(steel_defect_deploy_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

option(ENABLE_ORT "Enable ONNX Runtime backend" ON)
option(ENABLE_MNN "Enable MNN backend" OFF)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(STEEL_RPATHS "")

if(ENABLE_ORT)
    set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../ort_cpp/onnxruntime-linux-x64-1.23.2"
        CACHE PATH "Path to ONNX Runtime C++ SDK root")
    if(NOT EXISTS "${ORT_ROOT}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR "ONNX Runtime header not found at: ${ORT_ROOT}/include/onnxruntime_cxx_api.h")
    endif()
    if(NOT EXISTS "${ORT_ROOT}/lib/libonnxruntime.so")
        message(FATAL_ERROR "ONNX Runtime library not found at: ${ORT_ROOT}/lib/libonnxruntime.so")
    endif()
    list(APPEND STEEL_RPATHS "${ORT_ROOT}/lib")
endif()

if(ENABLE_MNN)
    set(MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/MNN"
        CACHE PATH "Path to MNN SDK or source+build root")
    if(NOT EXISTS "${MNN_ROOT}/include/MNN/Interpreter.hpp")
        message(FATAL_ERROR "MNN header not found at: ${MNN_ROOT}/include/MNN/Interpreter.hpp")
    endif()
    find_library(MNN_LIB NAMES MNN
        PATHS "${MNN_ROOT}/lib" "${MNN_ROOT}/build"
        NO_DEFAULT_PATH REQUIRED)
    get_filename_component(MNN_LIB_DIR "${MNN_LIB}" DIRECTORY)
    list(APPEND STEEL_RPATHS "${MNN_LIB_DIR}")
endif()

set(STEEL_CORE_SOURCES
    src/preprocess.cpp
    src/postprocess.cpp
    src/yolov5_lite_decoder.cpp
    src/frame_source.cpp
    src/async_pipeline.cpp
)

if(ENABLE_ORT)
    list(APPEND STEEL_CORE_SOURCES
        src/ort_yolov8_detector.cpp
        src/ort_yolov5lite_detector.cpp
    )
endif()

if(ENABLE_MNN)
    list(APPEND STEEL_CORE_SOURCES src/mnn_yolov5lite_detector.cpp)
endif()

add_library(steel_core STATIC ${STEEL_CORE_SOURCES})
target_include_directories(steel_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_compile_options(steel_core PRIVATE -Wall -Wextra)
target_link_libraries(steel_core PUBLIC
    ${OpenCV_LIBS}
    Threads::Threads
)

if(ENABLE_ORT)
    target_include_directories(steel_core PUBLIC "${ORT_ROOT}/include")
    target_link_directories(steel_core PUBLIC "${ORT_ROOT}/lib")
    target_link_libraries(steel_core PUBLIC onnxruntime)
    target_compile_definitions(steel_core PUBLIC USE_ORT)
endif()

if(ENABLE_MNN)
    target_include_directories(steel_core PUBLIC "${MNN_ROOT}/include")
    target_link_libraries(steel_core PUBLIC "${MNN_LIB}")
    target_compile_definitions(steel_core PUBLIC USE_MNN)
endif()

add_executable(steel_detect src/main.cpp)
target_compile_options(steel_detect PRIVATE -Wall -Wextra)
target_link_libraries(steel_detect PRIVATE steel_core)

enable_testing()
add_executable(steel_tests tests/test_core.cpp)
target_compile_options(steel_tests PRIVATE -Wall -Wextra)
target_link_libraries(steel_tests PRIVATE steel_core)
add_test(NAME steel_core_tests COMMAND steel_tests)

if(STEEL_RPATHS)
    set_target_properties(steel_detect PROPERTIES BUILD_RPATH "${STEEL_RPATHS}")
    set_target_properties(steel_tests PROPERTIES BUILD_RPATH "${STEEL_RPATHS}")
endif()
