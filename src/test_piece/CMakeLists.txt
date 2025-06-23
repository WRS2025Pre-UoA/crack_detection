cmake_minimum_required(VERSION 3.5)
project(test_piece)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === ONNX Runtime ===
# 🔽 解凍したtgzがある場所に合わせてパスを修正
set(ONNXRUNTIME_DIR "onnxruntime-linux-x64-1.21.0")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# === 自前ヘッダ ===
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/nn
)

file(GLOB_RECURSE CURR_SOURCES src/*.cpp)
add_executable(${PROJECT_NAME} ${CURR_SOURCES})

find_package(OpenCV REQUIRED)
find_package(ament_cmake REQUIRED)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  onnxruntime
)

set_target_properties(test_piece PROPERTIES 
 INSTALL_RPATH "$(ONNXRUNTIME_DIR)"
 BUILD_RPATH "$(ONNXRUNTIME_DIR)"
)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
