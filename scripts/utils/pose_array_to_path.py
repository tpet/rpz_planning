#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path


class PathConverter:
    """
    This ROS node subscribes to geometry_msgs/PoseArray topic,
    transforms this msg to nav_msgs/Path and publishes it to a separate topic
    """
    def __init__(self):
        self.path_pub = rospy.Publisher(rospy.get_param('~path_topic', 'path'), Path, queue_size=1)
        self.pose_arr_sub = rospy.Subscriber(rospy.get_param('~pose_array_topic', 'pci_command_path'), PoseArray,
                                             callback=self.path_cb, queue_size=1)

    def path_cb(self, pose_arr):
        assert isinstance(pose_arr, PoseArray)
        path_msg = Path()
        path_msg.header = pose_arr.header
        path_msg.poses = [PoseStamped(Header(), p) for p in pose_arr.poses]
        self.path_pub.publish(path_msg)


if __name__ == '__main__':
    rospy.init_node('path_converter')
    proc = PathConverter()
    rospy.spin()
