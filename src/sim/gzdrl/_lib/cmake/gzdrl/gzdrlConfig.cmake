# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was gzdrlConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# Dependency package files may reuse CMakePackageConfigHelpers' global
# PACKAGE_PREFIX_DIR variable. Preserve ours so this config stays relocatable.
set(_gzdrl_PACKAGE_PREFIX_DIR "${PACKAGE_PREFIX_DIR}")

# gzdrl CMake Config File
# 
# This file is used by find_package(gzdrl) to locate the gzdrl libraries
# and their dependencies.
#
# Usage:
#   find_package(gzdrl REQUIRED)
#   target_link_libraries(my_target gzdrl::rl_server gzdrl::rosrl_sitl)
#
# Exported targets:
#   gzdrl::rl_server     - Main RL server library
#   gzdrl::rosrl_sitl    - ROS SITL server library (if BUILD_SITL_ROS was ON)
#   gzdrl::uav_controllers - UAV controllers library
#   gzdrl::sensor_interface - Sensor interface library

include(CMakeFindDependencyMacro)

# Find required dependencies
find_dependency(Eigen3 REQUIRED)

# Find Gazebo dependencies
find_package(gz-cmake QUIET)
if (NOT gz-cmake_FOUND)
   # Fallback for older os 
   # pkg-config for older gazebo versions are not properly setup
   # Only gz-ionic is supported for fallback
   find_package(PkgConfig REQUIRED)
   pkg_check_modules(GZ_CMAKE QUIET gz-cmake4)
   if (GZ_CMAKE_FOUND)
     find_package(gz-cmake4 REQUIRED)
   else()
       message(FATAL_ERROR "gz-cmake4 package not found. For Ubuntu 22.04 and less, only gz-ionic is supported")
   endif()
   gz_find_package(gz-sim9 COMPONENTS
      gui
      rendering
      )
   gz_find_package(gz-math8)
   gz_find_package(gz-rendering9)
   gz_find_package(gz-common6)
   gz_find_package(gz-plugin3 COMPONENTS
      register
      )
   gz_find_package(gz-sensors9 COMPONENTS
      camera
      depth_camera
      gpu_lidar
      lidar
      rgbd_camera
      segmentation_camera
      thermal_camera
      wide_angle_camera
      )
   gz_find_package(sdformat15 
      )

    # Set version variables for fallback packages
    set(gz_sim_VERSION_MAJOR 9)
    set(gz_common_VERSION_MAJOR 6)
    set(gz_rendering_VERSION_MAJOR 9)
    set(gz_plugin_VERSION_MAJOR 3)
    set(gz_sensors_VERSION_MAJOR 9)
    set(sdformat_VERSION_MAJOR 15)

else()
   gz_find_package(gz-sim COMPONENTS
   gui
   rendering
   )
   gz_find_package(gz-math)
   gz_find_package(gz-rendering)
   gz_find_package(gz-common)
   gz_find_package(gz-plugin COMPONENTS
      register
      )
   gz_find_package(gz-sensors COMPONENTS
      camera
      depth_camera
      gpu_lidar
      lidar
      rgbd_camera
      segmentation_camera
      thermal_camera
      wide_angle_camera
      )
   gz_find_package(sdformat 
      )
   gz_find_package(gz-transport)
endif()

# Restore the prefix after loading dependencies that may overwrite it.
set(PACKAGE_PREFIX_DIR "${_gzdrl_PACKAGE_PREFIX_DIR}")

# Check if ROS is available and which version
set(gzdrl_ROS_VER )
set(ROS_VER )
set(ROS_PKGS )
if(gzdrl_ROS_VER EQUAL 1 OR gzdrl_ROS_VER EQUAL 2)
   include("${CMAKE_CURRENT_LIST_DIR}/hybrid_ros.cmake")
endif()

# Include the exported targets
include("${CMAKE_CURRENT_LIST_DIR}/gzdrlTargets.cmake")

# Check that required components are available
check_required_components(gzdrl)

# Provide some helpful variables
set(gzdrl_FOUND TRUE)
set(gzdrl_VERSION 1.0.0)
set_and_check(gzdrl_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
set_and_check(gzdrl_LIBRARY_DIR "${PACKAGE_PREFIX_DIR}/.")
set_and_check(gzdrl_RESOURCE_DIR "${PACKAGE_PREFIX_DIR}/gzdrl/resources")
set_and_check(gzdrl_SDF_DIR "${PACKAGE_PREFIX_DIR}/gzdrl/resources/sdf")
set(gzdrl_PLUGIN_DIR "${gzdrl_LIBRARY_DIR}")
set(gzdrl_INSTALL_PREFIX "${PACKAGE_PREFIX_DIR}")
unset(_gzdrl_PACKAGE_PREFIX_DIR)
