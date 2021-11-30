#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker


if __name__ == "__main__":
    rospy.init_node('mesh_publisher')

    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "my_namespace"
    marker.id = 0
    marker.action = Marker.ADD
    marker.pose.position.x = 1
    marker.pose.position.y = 1
    marker.pose.position.z = 1
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 1
    marker.scale.y = 1
    marker.scale.z = 1
    marker.color.a = 1.0  # Don't forget to set the alpha!
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    # only if using a MESH_RESOURCE marker type:
    marker.type = Marker.MESH_RESOURCE
    marker.mesh_resource = "package://rpz_planning/data/meshes/simple_cave_01.dae"
    # marker.mesh_resource = "package://rpz_planning/data/meshes/artifacts/black_and_decker_cordless_drill.dae"

    rate = rospy.Rate(1)
    vis_pub = rospy.Publisher('/mesh_topic', Marker, queue_size=1)
    while not rospy.is_shutdown():
        vis_pub.publish(marker)
        print(marker.pose)
        rate.sleep()
