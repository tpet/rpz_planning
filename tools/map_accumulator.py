#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
from ros_numpy import msgify, numpify
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf2_ros


class MapAccumulator:
    """
    This ROS node is simply subscribes to local map topic with PointCloud2 msgs
    and concatenates the observations in one global PointCloud2 map.
    The observations are firstly transformed to robot's ground truth frame
    before being concatenated.
    """
    def __init__(self,):
        self.points = None
        self.points_received = False
        rospy.Subscriber('local_map', PointCloud2, self.pc_callback)
        self.pc_pub = rospy.Publisher('~map', PointCloud2, queue_size=1)
        self.map_frame = None
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)

    def pc_callback(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        # Transform the point cloud
        self.map_frame = pc_msg.header.frame_id
        try:
            transform = self.tf.lookup_transform('X1_ground_truth', 'X1', rospy.Time(0))
            pc_msg = do_transform_cloud(pc_msg, transform)
            pc_msg.header.frame_id = self.map_frame
        except tf2_ros.LookupException:
            rospy.logwarn('No transform between estimated robot pose and its ground truth')

        new_points = numpify(pc_msg)
        if not self.points_received:
            self.points = new_points
            self.points_received = True
        self.points = np.concatenate([self.points, new_points])

        map_msg = msgify(PointCloud2, self.points)
        map_msg.header = pc_msg.header

        self.pc_pub.publish(map_msg)


if __name__ == '__main__':
    rospy.init_node('map_accumulator')
    proc = MapAccumulator()
    rospy.spin()
