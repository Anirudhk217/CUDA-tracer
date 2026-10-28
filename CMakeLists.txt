cmake_minimum_required(VERSION 3.18)
project(cudatrace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(cudatrace SHARED
  src/tracer.cpp
  src/trace_writer.cpp
)
target_link_libraries(cudatrace PRIVATE CUDA::cupti)
target_compile_options(cudatrace PRIVATE -Wall -Wextra -fvisibility=hidden)
# Find libcupti next to the toolkit at runtime without LD_LIBRARY_PATH.
set_target_properties(cudatrace PROPERTIES
  BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR};${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64"
  INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR};${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/lib64")

option(CUDATRACE_BUILD_SAMPLES "Build the sample CUDA programs" ON)
if(CUDATRACE_BUILD_SAMPLES)
  enable_language(CUDA)
  add_executable(vector_add samples/vector_add.cu)
  # "native" builds for the GPU in this machine (CMake 3.24+).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set_target_properties(vector_add PROPERTIES CUDA_ARCHITECTURES native)
  endif()
endif()

install(TARGETS cudatrace LIBRARY DESTINATION lib)
install(PROGRAMS scripts/cudatrace DESTINATION bin)

option(CUDATRACE_BUILD_TESTS "Build tests (no GPU needed)" ON)
if(CUDATRACE_BUILD_TESTS)
  enable_testing()
  add_executable(writer_test tests/writer_test.cpp src/trace_writer.cpp)
  add_test(NAME writer_test COMMAND writer_test ${CMAKE_BINARY_DIR}/writer_test.json)
endif()
