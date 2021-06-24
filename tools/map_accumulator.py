#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
from pyquaternion import Quaternion
from timeit import default_timer as timer
import torch


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
        self.map_frame = None
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)
        rospy.Subscriber('local_map', PointCloud2, self.pc_callback)
        rospy.loginfo('Map accumulator node is ready.')

    @staticmethod
    def transform_cloud(points, trans, quat):
        """
        Transform points (3 x N) from robot frame into a pinhole camera frame
        """
        assert isinstance(points, np.ndarray)
        assert isinstance(trans, np.ndarray)
        assert isinstance(quat, Quaternion)
        assert len(points.shape) == 2
        assert points.shape[0] == 3
        assert trans.shape == (3, 1)
        points = points - trans
        R_inv = np.asarray(quat.normalised.inverse.rotation_matrix, dtype=np.float32)
        points = np.matmul(R_inv, points)
        return points

    def pc_callback(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Point cloud is received')
        # Transform the point cloud
        self.map_frame = pc_msg.header.frame_id
        try:
            t0 = timer()
            transform = self.tf.lookup_transform('X1_ground_truth', 'X1', rospy.Time(0))
        except tf2_ros.LookupException:
            rospy.logwarn('Map accumulator: No transform between estimated robot pose and its ground truth')
            return
        translation = np.array([transform.transform.translation.x,
                               transform.transform.translation.y,
                               transform.transform.translation.z]).reshape([3, 1])
        quat = Quaternion(x=transform.transform.rotation.x,
                          y=transform.transform.rotation.y,
                          z=transform.transform.rotation.z,
                          w=transform.transform.rotation.w)
        points = numpify(pc_msg)
        # remove inf points
        mask = np.isfinite(points['x']) & np.isfinite(points['y']) & np.isfinite(points['z'])
        points = points[mask]
        # pull out x, y, and z values
        points3 = np.zeros([3] + list(points.shape), dtype=np.float32)
        points3[0, ...] = points['x']
        points3[1, ...] = points['y']
        points3[2, ...] = points['z']
        points3 = self.transform_cloud(points3, translation, quat)
        new_points = np.zeros(points3.shape[1], dtype=[('x', np.float32),
                                                       ('y', np.float32),
                                                       ('z', np.float32)])
        new_points['x'] = points3[0, ...]
        new_points['y'] = points3[1, ...]
        new_points['z'] = points3[2, ...]
        # rospy.logdebug('Point cloud conversion took: %.3f s', timer() - t0)

        # accumulate points
        if not self.points_received:
            self.points = new_points
            self.points_received = True
        self.points = np.concatenate([self.points, new_points])
        rospy.logdebug('Point cloud accumulation took: %.3f s', timer() - t0)

        # convert to message and publish
        map_msg = msgify(PointCloud2, self.points)
        map_msg.header = pc_msg.header
        self.pc_pub.publish(map_msg)


if __name__ == '__main__':
    rospy.init_node('map_accumulator', log_level=rospy.INFO)
    proc = MapAccumulator()
    rospy.spin()
