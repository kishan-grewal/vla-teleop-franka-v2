cmake_minimum_required(VERSION 3.16)
project(franka_xr_teleop_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp CONFIG QUIET)
find_package(Franka CONFIG QUIET)
find_package(nlohmann_json CONFIG QUIET)
set(USE_PKGCONFIG_FRANKA OFF)
if(NOT Franka_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FRANKA QUIET franka)
  if(NOT FRANKA_FOUND)
    pkg_check_modules(FRANKA QUIET libfranka)
  endif()
  if(NOT FRANKA_FOUND)
    message(FATAL_ERROR
            "libfranka not found. Install libfranka 0.9.2 (or >=0.9.1,<0.10.0) and set "
            "Franka_DIR/CMAKE_PREFIX_PATH or pkg-config paths.")
  endif()
  set(USE_PKGCONFIG_FRANKA ON)
endif()

set(USE_PKGCONFIG_YAMLCPP OFF)
if(NOT yaml-cpp_FOUND AND NOT TARGET yaml-cpp)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(YAMLCPP QUIET yaml-cpp)
  if(NOT YAMLCPP_FOUND)
    message(FATAL_ERROR
            "yaml-cpp not found. Install libyaml-cpp-dev or provide CMake package paths.")
  endif()
  set(USE_PKGCONFIG_YAMLCPP ON)
endif()

set(XROBOTICS_SERVICE_ROOT "/opt/apps/roboticsservice" CACHE PATH
    "XRoboToolkit PC Service installation root (contains SDK/)")
set(XROBOTICS_SDK_ROOT "${XROBOTICS_SERVICE_ROOT}/SDK" CACHE PATH
    "XRoboToolkit SDK root directory (contains include/ and linux*/64/)")

find_path(PXREA_SDK_INCLUDE_DIR
  NAMES PXREARobotSDK.h
  HINTS
    "${XROBOTICS_SDK_ROOT}/include"
    "${XROBOTICS_SERVICE_ROOT}/SDK/include"
    "$ENV{XROBOTICS_SDK_ROOT}/include"
    "$ENV{XROBOTICS_SERVICE_ROOT}/SDK/include")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(_pxrea_lib_subdir "linux_aarch64/64")
else()
  set(_pxrea_lib_subdir "linux/64")
endif()

find_library(PXREA_SDK_LIBRARY
  NAMES PXREARobotSDK libPXREARobotSDK
  HINTS
    "${XROBOTICS_SDK_ROOT}/${_pxrea_lib_subdir}"
    "${XROBOTICS_SERVICE_ROOT}/SDK/${_pxrea_lib_subdir}"
    "${XROBOTICS_SDK_ROOT}/x64"
    "${XROBOTICS_SERVICE_ROOT}/SDK/x64"
    "${XROBOTICS_SDK_ROOT}/arm64"
    "${XROBOTICS_SERVICE_ROOT}/SDK/arm64"
    "$ENV{XROBOTICS_SDK_ROOT}/${_pxrea_lib_subdir}"
    "$ENV{XROBOTICS_SERVICE_ROOT}/SDK/${_pxrea_lib_subdir}"
    "$ENV{XROBOTICS_SDK_ROOT}/x64"
    "$ENV{XROBOTICS_SERVICE_ROOT}/SDK/x64"
    "$ENV{XROBOTICS_SDK_ROOT}/arm64"
    "$ENV{XROBOTICS_SERVICE_ROOT}/SDK/arm64"
    "${XROBOTICS_SDK_ROOT}/lib"
    "$ENV{XROBOTICS_SDK_ROOT}/lib")

if(NOT PXREA_SDK_INCLUDE_DIR OR NOT PXREA_SDK_LIBRARY)
  message(FATAL_ERROR
          "XRoboToolkit SDK not found.\n"
          "Expected PXREARobotSDK.h and libPXREARobotSDK under SDK paths.\n"
          "Set -DXROBOTICS_SERVICE_ROOT=/opt/apps/roboticsservice or "
          "-DXROBOTICS_SDK_ROOT=/path/to/SDK.")
endif()

if(NOT nlohmann_json_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    HINTS
      "/usr/include"
      "/usr/local/include"
      "${XROBOTICS_SERVICE_ROOT}/PXREARobotSDK"
      "${XROBOTICS_SERVICE_ROOT}/RoboticsService/PXREARobotSDK"
      "${CMAKE_SOURCE_DIR}/third_party/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK"
      "$ENV{XROBOTICS_SERVICE_ROOT}/PXREARobotSDK"
      "$ENV{XROBOTICS_SERVICE_ROOT}/RoboticsService/PXREARobotSDK")
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
            "nlohmann/json.hpp not found.\n"
            "Install nlohmann-json3-dev or provide include path containing nlohmann/json.hpp.")
  endif()
endif()

add_executable(franka_xr_teleop_bridge
  main.cpp
  config_loader.cpp
  xrobotics_source.cpp
  policy_action_source.cpp
  teleop_mapper.cpp
  teleop_state_machine.cpp
  gripper.cpp
  franka_controller.cpp
  safety.cpp
  observation_pub.cpp
  trace_logger.cpp
)

add_executable(franka_xr_teleop_offline_qcmd_validator
  offline_qcmd_validator.cpp
  config_loader.cpp
)

target_link_libraries(franka_xr_teleop_bridge PRIVATE Threads::Threads)
target_link_libraries(franka_xr_teleop_bridge PRIVATE Eigen3::Eigen)
target_include_directories(franka_xr_teleop_bridge PRIVATE ${PXREA_SDK_INCLUDE_DIR})
target_link_libraries(franka_xr_teleop_bridge PRIVATE ${PXREA_SDK_LIBRARY})

target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE Threads::Threads)
target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE Eigen3::Eigen)

if(nlohmann_json_FOUND)
  target_link_libraries(franka_xr_teleop_bridge PRIVATE nlohmann_json::nlohmann_json)
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE nlohmann_json::nlohmann_json)
else()
  target_include_directories(franka_xr_teleop_bridge PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
  target_include_directories(franka_xr_teleop_offline_qcmd_validator PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
endif()

if(TARGET yaml-cpp)
  target_link_libraries(franka_xr_teleop_bridge PRIVATE yaml-cpp)
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE yaml-cpp)
elseif(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(franka_xr_teleop_bridge PRIVATE yaml-cpp::yaml-cpp)
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE yaml-cpp::yaml-cpp)
elseif(USE_PKGCONFIG_YAMLCPP)
  target_include_directories(franka_xr_teleop_bridge PRIVATE ${YAMLCPP_INCLUDE_DIRS})
  target_link_directories(franka_xr_teleop_bridge PRIVATE ${YAMLCPP_LIBRARY_DIRS})
  target_link_libraries(franka_xr_teleop_bridge PRIVATE ${YAMLCPP_LIBRARIES})
  target_compile_options(franka_xr_teleop_bridge PRIVATE ${YAMLCPP_CFLAGS_OTHER})
  target_include_directories(franka_xr_teleop_offline_qcmd_validator PRIVATE ${YAMLCPP_INCLUDE_DIRS})
  target_link_directories(franka_xr_teleop_offline_qcmd_validator PRIVATE ${YAMLCPP_LIBRARY_DIRS})
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE ${YAMLCPP_LIBRARIES})
  target_compile_options(franka_xr_teleop_offline_qcmd_validator PRIVATE ${YAMLCPP_CFLAGS_OTHER})
else()
  message(FATAL_ERROR "Could not find a known yaml-cpp CMake target.")
endif()

if(NOT USE_PKGCONFIG_FRANKA)
  if(DEFINED Franka_VERSION)
    if(Franka_VERSION VERSION_LESS 0.9.1 OR Franka_VERSION VERSION_GREATER_EQUAL 0.10.0)
      message(FATAL_ERROR
              "Unsupported libfranka version ${Franka_VERSION}. Expected >=0.9.1 and <0.10.0.")
    endif()
  endif()
endif()

if(TARGET Franka::Franka)
  target_link_libraries(franka_xr_teleop_bridge PRIVATE Franka::Franka)
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE Franka::Franka)
elseif(TARGET franka::franka)
  target_link_libraries(franka_xr_teleop_bridge PRIVATE franka::franka)
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE franka::franka)
elseif(TARGET franka)
  target_link_libraries(franka_xr_teleop_bridge PRIVATE franka)
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE franka)
elseif(USE_PKGCONFIG_FRANKA)
  target_include_directories(franka_xr_teleop_bridge PRIVATE ${FRANKA_INCLUDE_DIRS})
  target_link_directories(franka_xr_teleop_bridge PRIVATE ${FRANKA_LIBRARY_DIRS})
  target_link_libraries(franka_xr_teleop_bridge PRIVATE ${FRANKA_LIBRARIES})
  target_compile_options(franka_xr_teleop_bridge PRIVATE ${FRANKA_CFLAGS_OTHER})
  target_include_directories(franka_xr_teleop_offline_qcmd_validator PRIVATE ${FRANKA_INCLUDE_DIRS})
  target_link_directories(franka_xr_teleop_offline_qcmd_validator PRIVATE ${FRANKA_LIBRARY_DIRS})
  target_link_libraries(franka_xr_teleop_offline_qcmd_validator PRIVATE ${FRANKA_LIBRARIES})
  target_compile_options(franka_xr_teleop_offline_qcmd_validator PRIVATE ${FRANKA_CFLAGS_OTHER})
  if(DEFINED FRANKA_VERSION)
    if(FRANKA_VERSION VERSION_LESS 0.9.1 OR FRANKA_VERSION VERSION_GREATER_EQUAL 0.10.0)
      message(FATAL_ERROR
              "Unsupported libfranka version ${FRANKA_VERSION}. Expected >=0.9.1 and <0.10.0.")
    endif()
  endif()
else()
  message(FATAL_ERROR "Could not find a known libfranka CMake target.")
endif()
