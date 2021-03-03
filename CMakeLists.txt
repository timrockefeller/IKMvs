cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(PROJECT_NAME "IKMvs")
set(PROJECT_VERSION 0.0.1)

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(IKInit)

IK_AddPackage(IKit 0.0.3)

# OpenCV
message(STATUS "+ Fetching [OpenCV] from $ENV{OpenCV_DIR}.")
set(OpenCV_DIR $ENV{OpenCV_DIR})
find_package( OpenCV REQUIRED )
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "Package [OpenCV] not found.")
endif()

IK_InitProject(${PROJECT_NAME})

message(STATUS "[Project] ${PROJECT_NAME} ${PROJECT_VERSION}")

IK_AddSubDirsRec(include)
IK_AddSubDirsRec(src)