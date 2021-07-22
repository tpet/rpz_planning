#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros


class TravelledDistPub:
    """
    This ROS node publishes ground truth travelled distance.
    """

    def __init__(self):
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)
        self.rate = 10
        self.robot = rospy.get_param('robot', 'X1')
        self.world_frame = rospy.get_param('world_frame', 'subt')

        self.travelled_dist = 0.0
        self.robot_position = None
        self.initialized_pose = False

        self.dist_pub = rospy.Publisher('~travelled_dist', Float64, queue_size=1)

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            # travelled distance computation
            try:
                transform = self.tf.lookup_transform(self.world_frame, self.robot+'_ground_truth', rospy.Time(0))
                T = numpify(transform.transform)
                robot_position = T[:3, 3]
                if not self.initialized_pose:
                    self.robot_position = robot_position
                    self.initialized_pose = True

                self.travelled_dist += np.linalg.norm(self.robot_position - robot_position)
                self.robot_position = robot_position
                rospy.logdebug('Travelled distance: %.1f', self.travelled_dist)
                self.dist_pub.publish(Float64(self.travelled_dist))
            except tf2_ros.LookupException:
                rospy.logwarn('Robot ground truth pose is not available')
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('travelled_dist_publisher', log_level=rospy.INFO)
    proc = TravelledDistPub()
    proc.run()
