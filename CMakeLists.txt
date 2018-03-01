cmake_minimum_required(VERSION 2.8.3)
project(project)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DEIGEN_HAS_SFINAE)

find_package(OpenCV REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenPose REQUIRED)
find_package(Caffe REQUIRED)

add_definitions(-DOPENPOSE_BINARY_DIR="${CMAKE_BINARY_DIR}")
add_definitions(-DOPENPOSE_INSTALL_DIR="${CMAKE_INSTALL_PREFIX}")
add_definitions(-DCMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_definitions(-DCMAKE_MODELS="${CMAKE_SOURCE_DIR}/models/")


include_directories(${OpenPose_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${GLOG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${GLUT_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIRS}
  "/usr/include/GLFW/"
)
file(GLOB test_SRC
    "renderer.cpp"
    "renderer.h"
    "tensor.h"
    "trackbar.h"
    "smpl.h"
    "pf.h"
    "pf.cpp"
    "op.h"
    "test_np.py"
    "smplify_public/code/*.py"
    "smplify_public/code/smpl_webuser/*.py"
    "smplify_public/code/smpl_webuser/hello_world/*.py"
    "smplify_public/code/lib/*.py"
)

add_executable(test "test.cpp" ${test_SRC})
target_link_libraries(
  test
  ${OpenCV_LIBRARIES}
  ${GLUT_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY}
  jsoncpp
  "/usr/lib/x86_64-linux-gnu/libglfw.so.3.1"
)

add_executable(test2 "test2.cpp" "pf.h" "pf.cpp")
target_link_libraries(
  test2
  ${OpenCV_LIBRARIES}
  ${GLUT_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  jsoncpp
  "/usr/lib/x86_64-linux-gnu/libglfw.so.3.1"
)

add_executable(smpl_fit "smpl_fit.cpp" "pf.h" ${test_SRC})
target_link_libraries(
  smpl_fit
  ${OpenCV_LIBRARIES}
  ${GLUT_LIBRARY}
  ${OPENGL_LIBRARIES}
  ${GLEW_LIBRARIES}
  ${OpenPose_LIBS}
  jsoncpp
  "/usr/lib/x86_64-linux-gnu/libglfw.so.3.1"
)
