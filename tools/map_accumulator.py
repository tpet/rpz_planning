#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
from timeit import default_timer as timer
import torch


def transform_points(points, T):
    assert isinstance(points, np.ndarray)
    assert len(points.shape) == 2
    assert points.shape[0] >= 3  # (3, N) or (4, N)
    assert isinstance(T, np.ndarray)
    # Transform local map to ground truth localization frame
    R, t = T[:3, :3], T[:3, 3]
    points[:3, ...] = np.matmul(R, points[:3, ...]) + t.reshape([3, 1])
    return points


class MapAccumulator:
    """
    This ROS node is simply subscribes to local map topic with PointCloud2 msgs
    and concatenates the observations in one global PointCloud2 map.
    The observations are firstly transformed to robot's ground truth frame
    before being concatenated.
    """
    def __init__(self,):
        # Set the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.points = None
        self.points_received = False

        self.pc_pub = rospy.Publisher('~map', PointCloud2, queue_size=1)
        self.map_frame = rospy.get_param('~target_frame', 'world')
        self.cloud_rate = rospy.get_param('~cloud_rate', 1.0)

        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(100))
        self.tl = tf2_ros.TransformListener(self.tf)

        rospy.Subscriber(rospy.get_param('~local_map', '/X1/updated_map'), PointCloud2, self.pc_callback)
        rospy.loginfo('Map accumulator node is ready.')

    def pc_callback(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Point cloud is received')
        t0 = timer()

        # Transform the point cloud
        delay = (rospy.Time.now() - pc_msg.header.stamp).to_sec()
        rospy.logdebug('New points delay: %.3f s', delay)

        try:
            transform1 = self.tf.lookup_transform('X1', pc_msg.header.frame_id, pc_msg.header.stamp, rospy.Duration(3))
        except tf2_ros.LookupException:
            rospy.logwarn('Map accumulator: No transform between estimated robot pose and its ground truth')
            return

        try:
            transform2 = self.tf.lookup_transform(pc_msg.header.frame_id, 'X1_ground_truth', pc_msg.header.stamp, rospy.Duration(3))
        except tf2_ros.LookupException:
            rospy.logwarn('Map accumulator: No transform between estimated robot pose and its ground truth')
            return

        points = numpify(pc_msg)
        points = np.vstack([points[f] for f in ['x', 'y', 'z']])
        T1 = numpify(transform1.transform)
        T2 = numpify(transform2.transform)
        points = transform_points(points, np.matmul(T2, T1))
        new_points = np.zeros(points.shape[1], dtype=[('x', np.float32),
                                                      ('y', np.float32),
                                                      ('z', np.float32)])
        new_points['x'] = points[0, ...]
        new_points['y'] = points[1, ...]
        new_points['z'] = points[2, ...]
        # rospy.logdebug('Point cloud conversion took: %.3f s', timer() - t0)

        # accumulate points
        if not self.points_received:
            self.points = new_points
            self.points_received = True
        self.points = np.concatenate([self.points, new_points])
        self.map_frame = pc_msg.header.frame_id
        rospy.logdebug('Point cloud accumulation took: %.3f s', timer() - t0)

    def run(self):
        rate = rospy.Rate(self.cloud_rate)
        while not rospy.is_shutdown():
            if self.points is None:
                continue
            # convert to message and publish
            map_msg = msgify(PointCloud2, self.points)
            map_msg.header.frame_id = self.map_frame
            map_msg.header.stamp = rospy.Time.now()
            self.pc_pub.publish(map_msg)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('map_accumulator', log_level=rospy.INFO)
    proc = MapAccumulator()
    proc.run()
