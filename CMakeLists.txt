cmake_minimum_required(VERSION 3.11)

set(PROJECT_NAME "IKMvs")
set(PROJECT_VERSION 0.0.1)

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(IKInit)

IK_AddPackage(IKit v0.0.1)

IK_InitProject(${PROJECT_NAME})

message(STATUS "[Project] ${PROJECT_NAME} ${PROJECT_VERSION}")

include_directories (
    ${PROJECT_SOURCE_DIR}/include
    # ${CMAKE_BINARY_DIR}/config
    # $ENV{XXX_INCLUDE_PATH}
    # ...
)

link_directories (
    ${PROJECT_SOURCE_DIR}/lib
    # ${PROJECT_SOURCE_DIR}/lib/Gen
    # $ENV{XXX_LIB_PATH}
    # ...
)

IK_AddSubDirsRec(src)
