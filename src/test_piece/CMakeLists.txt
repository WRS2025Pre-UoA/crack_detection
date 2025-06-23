cmake_minimum_required(VERSION 3.5)
project(test_piece)

# === Project Settings ===
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Paths ===
set(ONNXRUNTIME_DIR "/home/user/onnxruntime-linux-x64-1.21.0")
include_directories(
  ${ONNXRUNTIME_DIR}/include
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/nn
)

# === Source Files ===
file(GLOB_RECURSE CURR_SOURCES src/*.cpp)
add_executable(${PROJECT_NAME} ${CURR_SOURCES})

# === Dependencies ===
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(ament_cmake REQUIRED)

# === Linking ===
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${realsense2_LIBRARY}
)

if(UNIX)
  target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

# === Compiler Warnings ===
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# === Testing and Linting ===
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Uncomment these if necessary
  # set(ament_cmake_cpplint_FOUND TRUE)
  # set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# === Install ===
install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
