cmake_minimum_required(VERSION 3.18)
project(torchtraincpp_cifar10 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(torchtraincpp
  src/main.cpp
  src/data/cifar10_dataset.cpp
  src/model/simple_cnn.cpp
  src/util/csv_logger.cpp
  src/util/checkpoint.cpp
)

target_include_directories(torchtraincpp PRIVATE src)
target_link_libraries(torchtraincpp "${TORCH_LIBRARIES}")

if(MSVC)
  target_compile_options(torchtraincpp PRIVATE /W4)
else()
  target_compile_options(torchtraincpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- RPATH configuration (Linux) ----
if(UNIX AND NOT APPLE)
  set_target_properties(torchtraincpp PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
endif()
