cmake_minimum_required(VERSION 3.19)
project(tokenizers_tvm_ffi LANGUAGES CXX C)

# tokenizers-cpp produces a STATIC library; PIC is required to link it into our SHARED wrapper.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
  OUTPUT_VARIABLE tvm_ffi_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _tvm_ffi_cmakedir_status
)
if(NOT _tvm_ffi_cmakedir_status EQUAL 0)
  message(FATAL_ERROR
    "apache-tvm-ffi was not found.  Install the Python package."
  )
endif()

find_package(tvm_ffi CONFIG REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tokenizers-cpp/CMakeLists.txt")
  message(FATAL_ERROR
    "Missing 3rdparty/tokenizers-cpp. "
    "Run: git submodule update --init --recursive"
  )
endif()

# tokenizers-cpp
add_subdirectory(3rdparty/tokenizers-cpp EXCLUDE_FROM_ALL)

add_library(tokenizers_tvm_ffi SHARED
  src/tokenizers_tvm_ffi/tokenizer.cc
)

target_include_directories(tokenizers_tvm_ffi PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tokenizers_tvm_ffi PRIVATE
  tokenizers_cpp
)

tvm_ffi_configure_target(
  tokenizers_tvm_ffi
  STUB_DIR "./python"
  STUB_INIT ON
  STUB_PKG "tokenizers_tvm_ffi"
  STUB_PREFIX "tokenizers_tvm_ffi."
)

install(TARGETS tokenizers_tvm_ffi DESTINATION .)
tvm_ffi_install(tokenizers_tvm_ffi DESTINATION .)
