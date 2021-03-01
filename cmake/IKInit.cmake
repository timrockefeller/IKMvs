message(STATUS "import IKInit.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/IKConfig.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/IKPackage.cmake")
set(IKCMAKE_VERSION 0.1.0)

macro(IK_InitProject PROJECT_NAME_STR)
message(STATUS "[IKCmake] ${IKCMAKE_VERSION}")
set(PROJECT_NAME ${PROJECT_NAME_STR})
IK_InitConfig()
# The version number.
set (${PROJECT_NAME}_VERSION_MAJOR 1)
set (${PROJECT_NAME}_VERSION_MINOR 0)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # 添加c++11标准支持
set (EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin") # 可执行文件输出目录

# CPack
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


endmacro(IK_InitProject PROJECT_NAME_STR)


#---------------------------------------
macro(IK_SetupProject MODE TARGET_NAME STR_TARGET_SOURCES STR_TARGET_LIBS)
    string(REPLACE " " ";" LIST_TARGET_SOURCES ${STR_TARGET_SOURCES})
    string(REPLACE " " ";" LIST_TARGET_LIBS ${STR_TARGET_LIBS})
    if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    endif(COMMAND cmake_policy)
    if(${STR_TARGET_SOURCES} STREQUAL " ")
        message(WARNING "Target [${TARGET_NAME}] has no source, so it won't be generated.")
    else()
        if(${MODE} STREQUAL "EXE")
            add_executable( ${TARGET_NAME} ${LIST_TARGET_SOURCES})
            set(INSTALL_DIR "bin")
            install (TARGETS ${TARGET_NAME} DESTINATION ${INSTALL_DIR})
        elseif(${MODE} STREQUAL "LIB")
            add_library(${TARGET_NAME} ${LIST_TARGET_SOURCES})
            #set(INSTALL_DIR "lib/Gen")
        else()
            message(FATAL_ERROR "Mode [${MODE}] is not supported, so target [TARGET_NAME] is not generated!")
            set(MODE_NOTSUPPORT " ")
        endif()
        if(NOT DEFINED MODE_NOTSUPPORT)
            if( NOT ${FOLDER_NAME} STREQUAL " ")
                SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES FOLDER ${FOLDER_NAME})
            endif()
            if(NOT ${STR_TARGET_LIBS} STREQUAL " ")
                target_link_libraries( ${TARGET_NAME} ${LIST_TARGET_LIBS} )
            endif()
            #install (TARGETS ${TARGET_NAME} DESTINATION ${INSTALL_DIR})
            message(STATUS "Setup Target ${FOLDER_NAME}/[${TARGET_NAME}] success")
        endif()
    endif()
endmacro(IK_SetupProject TARGET_NAME STR_TARGET_SOURCES STR_TARGET_LIBS)
