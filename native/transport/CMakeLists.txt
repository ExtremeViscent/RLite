cmake_minimum_required(VERSION 3.24)

project(rlite_transport_native VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "rlite_transport_native targets Linux only")
endif()

set(RLITE_TRANSPORT_LIBFABRIC_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/libfabric"
    CACHE PATH "Path to the vendored libfabric source tree"
)
set(RLITE_TRANSPORT_GDRCOPY_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../third-party/gdrcopy"
    CACHE PATH "Path to the vendored GDRCopy source tree"
)
set(RLITE_TRANSPORT_CUDA_INCLUDE_DIR
    ""
    CACHE PATH "Optional CUDA include directory; when empty the build falls back to stub CUDA declarations"
)
set(RLITE_TRANSPORT_NATIVE_OUTPUT_DIR
    ""
    CACHE PATH "Optional output directory for the built transport shared library"
)

if(NOT EXISTS "${RLITE_TRANSPORT_LIBFABRIC_SOURCE_DIR}/include/rdma/fabric.h")
    message(FATAL_ERROR "libfabric headers not found at ${RLITE_TRANSPORT_LIBFABRIC_SOURCE_DIR}")
endif()

if(NOT EXISTS "${RLITE_TRANSPORT_GDRCOPY_SOURCE_DIR}/include/gdrapi.h")
    message(FATAL_ERROR "GDRCopy headers not found at ${RLITE_TRANSPORT_GDRCOPY_SOURCE_DIR}")
endif()

add_library(rlite_transport_native SHARED
    src/rlite_transport_native.cpp
)

target_compile_features(rlite_transport_native PRIVATE cxx_std_17)
target_compile_definitions(rlite_transport_native PRIVATE _GNU_SOURCE)
target_include_directories(rlite_transport_native
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${RLITE_TRANSPORT_LIBFABRIC_SOURCE_DIR}/include"
        "${RLITE_TRANSPORT_GDRCOPY_SOURCE_DIR}/include"
)

if(RLITE_TRANSPORT_CUDA_INCLUDE_DIR)
    target_include_directories(rlite_transport_native PRIVATE "${RLITE_TRANSPORT_CUDA_INCLUDE_DIR}")
else()
    foreach(cuda_candidate IN ITEMS
        "/usr/local/cuda/include"
        "/opt/cuda/include"
        "/usr/include"
    )
        if(EXISTS "${cuda_candidate}/cuda_runtime_api.h")
            target_include_directories(rlite_transport_native PRIVATE "${cuda_candidate}")
            break()
        endif()
    endforeach()
endif()

target_link_libraries(rlite_transport_native PRIVATE dl pthread)
set_target_properties(rlite_transport_native PROPERTIES
    OUTPUT_NAME "rlite_transport_native"
    POSITION_INDEPENDENT_CODE ON
)

if(RLITE_TRANSPORT_NATIVE_OUTPUT_DIR)
    set_target_properties(rlite_transport_native PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${RLITE_TRANSPORT_NATIVE_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${RLITE_TRANSPORT_NATIVE_OUTPUT_DIR}"
        ARCHIVE_OUTPUT_DIRECTORY "${RLITE_TRANSPORT_NATIVE_OUTPUT_DIR}"
    )
endif()
