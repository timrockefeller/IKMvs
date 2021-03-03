#---------------------------------------
macro(IK_SetupProject MODE TARGET_NAME STR_TARGET_SOURCES STR_TARGET_LIBS)
  IK_PackageName(package_name)
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
      message(STATUS "Setup Target ${FOLDER_NAME}/[${TARGET_NAME}] success")
    endif()
  endif()
endmacro()


function(IK_GlobGroupSrcs rst _sources)
	set(tmp_rst "")
	foreach(path ${${_sources}})
    if(IS_DIRECTORY ${path})
      file(GLOB_RECURSE pathSources
        ${path}/*.h
        ${path}/*.hpp
        ${path}/*.inl
        ${path}/*.c
        ${path}/*.cc
        ${path}/*.cpp
        ${path}/*.cxx
      )
      list(APPEND tmp_rst ${pathSources})
    else()
      if(NOT IS_ABSOLUTE "${path}")
        get_filename_component(path "${path}" ABSOLUTE)
      endif()
      list(APPEND tmp_rst ${path})
    endif()
  endforeach()
	set(${rst} ${tmp_rst} PARENT_SCOPE)
endfunction()


function(IK_GetTargetName out targetPath)
  file(RELATIVE_PATH targetRelPath "${PROJECT_SOURCE_DIR}/src" "${targetPath}")
  string(REPLACE "/" "_" target_name "${PROJECT_NAME}_${targetRelPath}")
  set(${out} ${target_name} PARENT_SCOPE)
endfunction()


function(IK_AddTarget)
  # [option]
  # TEST
  # NOT_HERE
  # [value]
  # MODE: EXE / STATIC / SHARED / INTERFACE
  # ADD_CURRENT_TO: PUBLIC / INTERFACE / PRIVATE (default) / NONE
  # TARGET_NAME
  # RET_TARGET_NAME
  # [list]
  # SRC: dir | file | current_dir(default)
  # INC: dir
  # LIB: <lib-target> | *.lib
  # C_OPTION: compile options
  # L_OPTION: link options
  message(STATUS "┌────────────────────────────────────────────────┐")
  set(arglist "")

  # publics
  list(APPEND arglist SRC_PUB INC LIB C_OPTION L_OPTION)
  # interfaces
  list(APPEND arglist SRC_INT INC_INT LIB_INT C_OPTION_INT L_OPTION_INT)
  # privates
  list(APPEND arglist SRC INC_PVT LIB_PVT C_OPTION_PVT L_OPTION_PVT)

  cmake_parse_arguments(
    "ARG"
    "TEST;NOT_HERE"
    "MODE;ADD_CURRENT_TO;OUTPUT_NAME;TARGET_NAME;RET_TARGET_NAME"
    "${arglist}"
    ${ARGN}
  )
  
  # default
  if("${ARG_ADD_CURRENT_TO}" STREQUAL "")
    set(ARG_ADD_CURRENT_TO "PRIVATE")
  endif()

  if("${ARG_MODE}" STREQUAL "INTERFACE")
    list(APPEND ARG_SRC_INT       ${ARG_SRC_PUB}      ${ARG_SRC}          )
    list(APPEND ARG_INC_INT       ${ARG_INC}          ${ARG_INC_PVT}      )
    list(APPEND ARG_LIB_INT       ${ARG_LIB}          ${ARG_LIB_PVT}      )
    list(APPEND ARG_C_OPTION_INT  ${ARG_C_OPTION}     ${ARG_C_OPTION_PVT} )
    list(APPEND ARG_L_OPTION_INT  ${ARG_L_OPTION}     ${ARG_L_OPTION_PVT} )
    set(ARG_SRC_PUB      "")
    set(ARG_SRC          "")
    set(ARG_INC          "")
    set(ARG_INC_PVT      "")
    set(ARG_LIB          "")
    set(ARG_LIB_PVT      "")
    set(ARG_C_OPTION     "")
    set(ARG_C_OPTION_PVT "")
    set(ARG_L_OPTION     "")
    set(ARG_L_OPTION_PVT "")
    if(NOT "${ARG_ADD_CURRENT_TO}" STREQUAL "NONE")
      set(ARG_ADD_CURRENT_TO "INTERFACE")
    endif()
  endif()

  # sources
  if("${ARG_ADD_CURRENT_TO}" STREQUAL "PUBLIC")
    list(APPEND ARG_SRC_PUB ${CMAKE_CURRENT_SOURCE_DIR})
  elseif("${ARG_ADD_CURRENT_TO}" STREQUAL "INTERFACE")
    list(APPEND ARG_SRC_INT ${CMAKE_CURRENT_SOURCE_DIR})
  elseif("${ARG_ADD_CURRENT_TO}" STREQUAL "PRIVATE")
    list(APPEND ARG_SRC ${CMAKE_CURRENT_SOURCE_DIR})
  elseif(NOT "${ARG_ADD_CURRENT_TO}" STREQUAL "NONE")
    message(FATAL_ERROR "ADD_CURRENT_TO [${ARG_ADD_CURRENT_TO}] is not supported")
  endif()

  IK_GlobGroupSrcs(sources_public ARG_SRC_PUB)
  IK_GlobGroupSrcs(sources_interface ARG_SRC_INT)
  IK_GlobGroupSrcs(sources_private ARG_SRC)

  if(NOT NOT_HERE)
    set(all_sources ${sources_public} ${sources_interface} ${sources_private})
    foreach(src ${allsources})
      get_filename_component(dir ${src} DIRECTORY)
      string(FIND ${dir} ${CMAKE_CURRENT_SOURCE_DIR} idx)
      if(NOT idx EQUAL -1)
        set(base_dir "${CMAKE_CURRENT_SOURCE_DIR}/..")
        file(RELATIVE_PATH rdir "${CMAKE_CURRENT_SOURCE_DIR}/.." ${dir})
      else()
        set(base_dir ${PROJECT_SOURCE_DIR})
      endif()
      file(RELATIVE_PATH rdir ${base_dir} ${dir})
      if(MSVC)
        string(REPLACE "/" "\\" rdir_MSVC ${rdir})
        set(rdir "${rdir_MSVC}")
      endif()
      source_group(${rdir} FILES ${src})
    endforeach()
  endif()

  # target folder
  file(RELATIVE_PATH targetRelPath "${PROJECT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/..")
  set(target_folder "${PROJECT_NAME}/${targetRelPath}")

  # target name
  if("${ARG_TARGET_NAME}" STREQUAL "")
    IK_GetTargetName(target_name ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    set(target_name ${ARG_TARGET_NAME})
  endif()
  if(NOT "${ARG_RET_TARGET_NAME}" STREQUAL "")
    set(${ARG_RET_TARGET_NAME} ${target_name} PARENT_SCOPE)
  endif()

  IK_PackageName(package_name)
  message(STATUS " Building target [${target_name}] in package [${package_name}]")
  # add target
  if("${ARG_MODE}" STREQUAL "EXE")
    add_executable(${target_name})
    add_executable("KTKR::${target_name}" ALIAS ${target_name})
    if(MSVC)
      set_target_properties(${target_name} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${IK_RootProjectPath}/bin")
    endif()
    set_target_properties(${target_name} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
  elseif("${ARG_MODE}" STREQUAL "STATIC")
    add_library(${target_name} STATIC)
    add_library("KTKR::${target_name}" ALIAS ${target_name})
  elseif("${ARG_MODE}" STREQUAL "SHARED")
    add_library(${target_name} SHARED)
    add_library("KTKR::${target_name}" ALIAS ${target_name})
  elseif("${ARG_MODE}" STREQUAL "INTERFACE")
    add_library(${target_name} INTERFACE)
    add_library("KTKR::${target_name}" ALIAS ${target_name})
  else()
    message(FATAL_ERROR "Unknown mode [${ARG_MODE}].")
    return()
  endif()

  # folder
  if(NOT ${ARG_MODE} STREQUAL "INTERFACE")
    set_target_properties(${target_name} PROPERTIES FOLDER ${target_folder})
  endif()

  # target source files
  foreach(src ${source_public})
    get_filename_component(abs_src ${src} ABSOLUTE)
    file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
    target_sources(${target_name} PUBLIC
      $<BUILD_INTERFACE:${abs_src}>
      $<INSTALL_INTERFACE:${package_name}/${rel_src}>
    )
  endforeach()
  foreach(src ${source_interface})
    get_filename_component(abs_src ${src} ABSOLUTE)
    file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
    target_sources(${target_name} INTERFACE
      $<BUILD_INTERFACE:${abs_src}>
      $<INSTALL_INTERFACE:${package_name}/${rel_src}>
    )
  endforeach()
  foreach(src ${sources_private})
    get_filename_component(abs_src ${src} ABSOLUTE)
    file(RELATIVE_PATH rel_src ${PROJECT_SOURCE_DIR} ${abs_src})
    target_sources(${target_name} PRIVATE
      $<BUILD_INTERFACE:${abs_src}>
      $<INSTALL_INTERFACE:${package_name}/${rel_src}>
    )
  endforeach()

  # target library files
  target_link_libraries(${target_name}
    PUBLIC ${ARG_LIB}
    INTERFACE ${ARG_LIB_INT}
    PRIVATE ${ARG_LIB_PVT}
  )

 # target include files
  foreach(inc ${ARG_INC})
    get_filename_component(abs_inc ${inc} ABSOLUTE)
    file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${abs_inc})
    target_include_directories(${target_name} PUBLIC
      $<BUILD_INTERFACE:${abs_inc}>
      $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
    )
  endforeach()
    foreach(inc ${ARG_INC_PVT})
    get_filename_component(abs_inc ${inc} ABSOLUTE)
    file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${abs_inc})
    target_include_directories(${target_name} PRIVATE
      $<BUILD_INTERFACE:${abs_inc}>
      $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
    )
  endforeach()
    foreach(inc ${ARG_INC_INT})
    get_filename_component(abs_inc ${inc} ABSOLUTE)
    file(RELATIVE_PATH rel_inc ${PROJECT_SOURCE_DIR} ${inc})
    target_include_directories(${target_name} INTERFACE
      $<BUILD_INTERFACE:${abs_inc}>
      $<INSTALL_INTERFACE:${package_name}/${rel_inc}>
    )
  endforeach()

  # target compile option
  target_compile_options(${target_name}
    PUBLIC ${ARG_C_OPTION}
    INTERFACE ${ARG_C_OPTION_INT}
    PRIVATE ${ARG_C_OPTION_PVT}
  )
  
  # target link option
  target_link_options(${target_name}
    PUBLIC ${ARG_L_OPTION}
    INTERFACE ${ARG_L_OPTION_INT}
    PRIVATE ${ARG_L_OPTION_PVT}
  )

  if(NOT "${ARG_OUTPUT_NAME}" STREQUAL "")
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME "${ARG_OUTPUT_NAME}" CLEAN_DIRECT_OUTPUT 1)
  endif()

  # export
  if(NOT ARG_TEST)
    install(
      TARGETS ${target_name}
      EXPORT "${PROJECT_NAME}Targets"
      RUNTIME DESTINATION "bin"
      ARCHIVE DESTINATION "${package_name}/lib"
      LIBRARY DESTINATION "${package_name}/lib"
    )
  endif()
  message(STATUS " Project: ${PROJECT_NAME}")
  message(STATUS "└────────────────────────────────────────────────┘")
endfunction()