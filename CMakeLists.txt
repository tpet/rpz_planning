cmake_minimum_required(VERSION 3.9)
project(rpz_planning)

# Catkin
set(CATKIN_DEPS
    geometry_msgs
    nav_msgs
    sensor_msgs
    std_msgs
    tf2_ros
    message_generation
)
find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

## Generate messages in the 'msg' folder
add_message_files(DIRECTORY msg FILES
  Metrics.msg
)

catkin_python_setup()

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
    CATKIN_DEPENDS ${CATKIN_DEPS}
)

install(
    PROGRAMS
        scripts/points_occlusion
        scripts/rpz_planner
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
    DIRECTORY
        launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

