
set(IK_${PROJECT_NAME}_havedep FALSE)

function(IK_AddSubDirsRec path)
  message(STATUS "----------")
  file(GLOB_RECURSE children LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/${path}/*)
  set(dirs "")
  list(APPEND children "${CMAKE_CURRENT_SOURCE_DIR}/${path}")
  foreach(item ${children})
    if(IS_DIRECTORY ${item} AND EXISTS "${item}/CMakeLists.txt")
      list(APPEND dirs ${item})
    endif()
  endforeach()
  foreach(dir ${dirs})
    add_subdirectory(${dir})
  endforeach()
endfunction()

### IK_UnityArgs
## (tarLIST STR1 STR2 ...)
## -> tarLIST = "STR1 STR2 ..."
function(IK_UnityArgs)
    set(ARGLIST "")
    set(IK_INDEX 1)
    if(${ARGC} LESS 2)
        set(ARGLIST " ")
    else()
        while(IK_INDEX LESS ${ARGC})
            set(ARGLIST "${ARGLIST} ${ARGV${IK_INDEX}}")
            math(EXPR IK_INDEX "${IK_INDEX} + 1")  
        endwhile()
    endif()
    set(${ARGV0} ${ARGLIST} PARENT_SCOPE)
endfunction(IK_UnityArgs)

function(IK_ThisToPackageName out name version)
  set(tmp "${name}.${version}")
  string(REPLACE "." "_" tmp ${tmp})
  set(${out} ${tmp} PARENT_SCOPE)
endfunction()

function(IK_PackageName out)
  IK_ThisToPackageName(tmp ${PROJECT_NAME} ${PROJECT_VERSION})
  set(${out} ${tmp} PARENT_SCOPE)
endfunction()

function(IK_AddPackage name version)
  # 添加第二方依赖包
  set(IK_${PROJECT_NAME}_havedep TRUE)
  list(FIND IK_${PROJECT_NAME}_dep_name_list "${name}" _idx)
  if(_idx EQUAL -1)
    message(STATUS "Start add dependence ${name} ${version}.")
    set(_need_fetch TRUE)
  else()
    set(_A_version "${${name}_VERSION}")
    set(_B_version "${version}")
    if(_A_version EQUAL _B_version)
      message(STATUS "Dependence's version in ${version} already fit.")
      set(_need_fetch FALSE)
      else()
      message(FATAL_ERROR "Dependence's version incapable with ${_A_version} and ${_B_version}.")
    endif()
  endif()
  if(_need_fetch)
    list(APPEND IK_${PROJECT_NAME}_dep_name_list ${name})
    list(APPEND IK_${PROJECT_NAME}_dep_version_list ${version})
    message(STATUS "find package: ${name} ${version}...")
    find_package(${name} ${version} QUIET)
    if(${${name}_FOUND})
      message(STATUS "OK: ${name} ${${name}_VERSION} found.")
    else()
    include(FetchContent)
      message(STATUS "- fetching ${name}...")
      set(IKIT_GIT_TAG  ${version})  # 指定版本
      set(IKIT_GIT_URL  "https://github.com/timrockefeller/${name}.git")  # 指定git仓库地址
      FetchContent_Declare(
        ${name}
        GIT_REPOSITORY    ${IKIT_GIT_URL}
        GIT_TAG           ${IKIT_GIT_TAG}
      )
      FetchContent_MakeAvailable(${name})
      message(STATUS "OK: ${name} ${version} fetched.")
    endif()
  endif()
endfunction()

macro (IK_Export)
  cmake_parse_arguments("ARG" "TARGET" "" "DIRECTORIES" ${ARGN})

  IK_PackageName(package_name)
  message(STATUS "export ${package_name}")

  set(IK_PACKAGE_INIT "
get_filename_component(include_dir \"\${CMAKE_CURRENT_LIST_DIR}/../include\" ABSOLUTE)
include_directories(\"\${include_dir}\")\n")

  if(ARG_TARGET)
    # generate the export targets for the build tree
    # needs to be after the install(TARGETS) command
    export(EXPORT "${PROJECT_NAME}Targets"
      NAMESPACE "KTKR::"
      #FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
    )
    
    # install the configuration targets
    install(EXPORT "${PROJECT_NAME}Targets"
      FILE "${PROJECT_NAME}Targets.cmake"
      NAMESPACE "KTKR::"
      DESTINATION "${package_name}/cmake"
    )
  endif()


  include(CMakePackageConfigHelpers)
  # generate the config file that is includes the exports
  configure_package_config_file(${PROJECT_SOURCE_DIR}/config/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION "${package_name}/cmake"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
  
  # generate the version file for the config file
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
  )

  # install the configuration file
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${package_name}/cmake"
  )
  
  set(${package_name}_DIR "${package_name}/cmake")

  foreach(dir ${ARG_DIRECTORIES})
    string(REGEX MATCH "(.*)/" prefix ${dir})
    if("${CMAKE_MATCH_1}" STREQUAL "")
      set(_destination "${package_name}")
    else()
      set(_destination "${package_name}/${CMAKE_MATCH_1}")
    endif()
    install(DIRECTORY ${dir} DESTINATION "${_destination}")
  endforeach()
  message(STATUS "OK: exported ${package_name}")
endmacro()