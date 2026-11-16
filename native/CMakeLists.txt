# CzechSpeech native library: whisper.cpp + ggml linked statically into one JNI library.
#
# One configure/build directory per CPU variant; native/build-native.sh drives it. The variant is chosen with the
# ggml options passed on the command line (GGML_CPU_ARM_ARCH for arm64, GGML_SSE42 for the x86_64 emulator,
# GGML_NATIVE for the host build used by JVM tests) and the output name with CZECHSPEECH_LIB_NAME.
cmake_minimum_required(VERSION 3.22)
project(czechspeech C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CZECHSPEECH_LIB_NAME "czechspeech" CACHE STRING "Output name of the JNI library, without the lib prefix")
set(WHISPER_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/whisper.cpp" CACHE PATH "whisper.cpp source tree")

if (NOT EXISTS "${WHISPER_CPP_DIR}/include/whisper.h")
    message(FATAL_ERROR "whisper.cpp not found in ${WHISPER_CPP_DIR}; run: git submodule update --init")
endif()

# whisper.cpp and ggml as static libraries, CPU backend only, no OpenMP, no dynamic backend loading.
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(WHISPER_CURL           OFF CACHE BOOL "" FORCE)
set(WHISPER_SDL2           OFF CACHE BOOL "" FORCE)
set(WHISPER_ALL_WARNINGS   OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP            OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL        OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS  OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE            OFF CACHE BOOL "" FORCE)
set(GGML_METAL             OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN            OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL            OFF CACHE BOOL "" FORCE)
set(GGML_CUDA              OFF CACHE BOOL "" FORCE)

add_subdirectory("${WHISPER_CPP_DIR}" whisper.cpp EXCLUDE_FROM_ALL)

add_library(czechspeech SHARED src/czechspeech_jni.c)
set_target_properties(czechspeech PROPERTIES
    OUTPUT_NAME "${CZECHSPEECH_LIB_NAME}"
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)
target_link_libraries(czechspeech PRIVATE whisper)

if (ANDROID)
    target_link_libraries(czechspeech PRIVATE log)
else()
    # Host build for JVM tests: JNI headers from the JDK.
    if (NOT DEFINED ENV{JAVA_HOME})
        message(FATAL_ERROR "JAVA_HOME must point to a JDK for the host build")
    endif()
    target_include_directories(czechspeech PRIVATE "$ENV{JAVA_HOME}/include" "$ENV{JAVA_HOME}/include/linux")
endif()

# Only the JNI entry points are exported; whisper/ggml symbols from the static archives stay hidden.
target_link_options(czechspeech PRIVATE
    -Wl,--gc-sections
    -Wl,--exclude-libs,ALL
    -Wl,--no-undefined)

if (ANDROID)
    # 16 KB page alignment of LOAD segments (required by Google Play for Android 15+ devices).
    target_link_options(czechspeech PRIVATE -Wl,-z,max-page-size=16384)
endif()
