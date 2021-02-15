cmake_minimum_required(VERSION 3.9)
project(rpz_planning)

# Catkin
set(CATKIN_DEPS
    geometry_msgs
    message_runtime
    nav_msgs
    sensor_msgs
    std_msgs
    tf2_ros
)
find_package(catkin REQUIRED COMPONENTS ${CATKIN_DEPS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

catkin_package(
    CATKIN_DEPENDS ${CATKIN_DEPS}
)

catkin_python_setup()

install(
    PROGRAMS
        scripts/rpz_planner
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
    DIRECTORY
        launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
