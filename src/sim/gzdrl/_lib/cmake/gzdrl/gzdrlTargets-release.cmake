#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gzdrl::uav_controllers" for configuration "Release"
set_property(TARGET gzdrl::uav_controllers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gzdrl::uav_controllers PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/./libuav_controllers.so"
  IMPORTED_SONAME_RELEASE "libuav_controllers.so"
  )

list(APPEND _cmake_import_check_targets gzdrl::uav_controllers )
list(APPEND _cmake_import_check_files_for_gzdrl::uav_controllers "${_IMPORT_PREFIX}/./libuav_controllers.so" )

# Import target "gzdrl::sensor_interface" for configuration "Release"
set_property(TARGET gzdrl::sensor_interface APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gzdrl::sensor_interface PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/./libsensor_interface.so"
  IMPORTED_SONAME_RELEASE "libsensor_interface.so"
  )

list(APPEND _cmake_import_check_targets gzdrl::sensor_interface )
list(APPEND _cmake_import_check_files_for_gzdrl::sensor_interface "${_IMPORT_PREFIX}/./libsensor_interface.so" )

# Import target "gzdrl::rl_server" for configuration "Release"
set_property(TARGET gzdrl::rl_server APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gzdrl::rl_server PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/./librl_server.so"
  IMPORTED_SONAME_RELEASE "librl_server.so"
  )

list(APPEND _cmake_import_check_targets gzdrl::rl_server )
list(APPEND _cmake_import_check_files_for_gzdrl::rl_server "${_IMPORT_PREFIX}/./librl_server.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
