#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
from rpz_planning.io import write, append
import os


class TravelledDistPub:
    """
    This ROS node publishes ground truth travelled distance and route.
    """

    def __init__(self):
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)
        self.rate = rospy.get_param('~rate', 10.0)  # rate = 10 -> dt = 0.1
        self.robot = rospy.get_param('robot', 'X1')
        self.world_frame = rospy.get_param('world_frame', 'subt')

        # travelled dist to publish
        self.travelled_dist = 0.0
        self.eps = 0.005
        self.robot_position = None
        self.initialized_pose = False

        # route to publish
        self.route = Path()
        self.route.header.frame_id = self.world_frame
        self.wps_dist = rospy.get_param('~route_wps_dist', 1.0)  # [m], dist between sampled path waypoints
        self.route_pub = rospy.Publisher('~route', Path, queue_size=2)

        self.dist_pub = rospy.Publisher('~travelled_dist', Float64, queue_size=2)
        self.poses_csv_path = rospy.get_param('~poses_gt', 'poses_gt.csv')
        if os.path.exists(self.poses_csv_path):
            rospy.logwarn('File with SLAM poses already exists: %s. It will be overwritten.',
                          os.path.realpath(self.poses_csv_path))
        write(self.poses_csv_path,
              'poseId, timestamp, T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23, T30, T31, T32, T33\n')
        self.pose_ind = 0

    def run(self):
        rate = rospy.Rate(self.rate)
        prev_wp = None
        while not rospy.is_shutdown():
            # travelled distance computation
            try:
                t_now = rospy.Time.now()
                transform = self.tf.lookup_transform(self.world_frame, self.robot+'_ground_truth',
                                                     rospy.Time(0), rospy.Duration(1))
                T = numpify(transform.transform)
                assert T.shape == (4, 4)
                append(self.poses_csv_path, '%d, %.6f, %s\n'
                       % (self.pose_ind, t_now.to_sec(), ', '.join(['%.6f' % x for x in T.flatten()])))
                self.pose_ind += 1

                prev_position = T[:3, 3]
                if not self.initialized_pose:
                    self.robot_position = prev_position
                    prev_wp = prev_position
                    self.initialized_pose = True

                # publish travelled distance so far
                dp = np.linalg.norm(self.robot_position - prev_position)
                dp = dp if dp > self.eps else 0.0  # do not add negligible movement
                self.travelled_dist += dp
                self.robot_position = prev_position
                self.dist_pub.publish(Float64(self.travelled_dist))

                # add waypoints every wps_dist to a route and publish it
                dwp = np.linalg.norm(self.robot_position - prev_wp)
                if dwp >= self.wps_dist:
                    rospy.logdebug('Travelled distance: %.1f', self.travelled_dist)

                    # append wp to path
                    pose = PoseStamped()
                    pose.header.frame_id = self.world_frame
                    pose.header.stamp = rospy.Time.now()
                    pose.pose.position.x = transform.transform.translation.x
                    pose.pose.position.y = transform.transform.translation.y
                    pose.pose.position.z = transform.transform.translation.z
                    pose.pose.orientation = transform.transform.rotation

                    self.route.poses.append(pose)
                    self.route.header.stamp = rospy.Time.now()
                    self.route_pub.publish(self.route)

                    prev_wp = self.robot_position

            except (tf2_ros.LookupException, rospy.exceptions.ROSTimeMovedBackwardsException):
                rospy.logwarn('Robot ground truth pose is not available')
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('travelled_dist_publisher', log_level=rospy.INFO)
    proc = TravelledDistPub()
    proc.run()
