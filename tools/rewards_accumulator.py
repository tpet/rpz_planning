#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
from ros_numpy import msgify, numpify
import tf2_ros

import torch
import numpy as np
from pytorch3d.ops.knn import knn_points
from timeit import default_timer as timer


def transform_points(points, transform):
    assert isinstance(points, torch.Tensor)
    assert len(points.shape) == 2
    assert points.shape[0] == 3 or points.shape[0] == 4  # (3, N) or (4, N)
    assert isinstance(transform, TransformStamped)
    # Transform local map to ground truth localization frame
    T = torch.as_tensor(numpify(transform.transform), dtype=torch.float32).to(points.device)
    if points.shape[1] == 3:
        R, t = T[:3, :3], T[:3, 3]
        points = torch.matmul(R, points) + t.reshape([3, 1])
    elif points.shape[0] == 4:
        points = torch.matmul(T, points)
    return points


class RewardsAccumulator:
    """
    This ROS node subscribes to local map rewards cloud with PointCloud2 msgs
    and concatenates the observations in one global PointCloud2 map.
    Merging the points is done by determining firstly the new points by
    proximity threshold to the points in existing map.
    The observations are firstly transformed to robot's ground truth frame
    before being concatenated.
    """
    def __init__(self, reward_cloud_topic='reward_cloud'):
        # Set the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.global_map = None
        self.map_frame = None
        self.new_points = None
        # any point that is farer than this threshold from points in existing pcl is considered as new
        self.dist_th = rospy.get_param('~pts_proximity_th', 0.5)
        self.max_age = rospy.get_param('~max_age', 10.0)
        self.reward_cloud_rate = rospy.get_param('~reward_cloud_rate', 0.2)
        self.rewards_cloud_pub = rospy.Publisher('~rewards_map', PointCloud2, queue_size=1)
        self.new_cloud_pub = rospy.Publisher('~new_rewards_map', PointCloud2, queue_size=1)
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)
        self.local_map_sub = rospy.Subscriber(reward_cloud_topic, PointCloud2, self.accumulate_reward_clouds_cb)
        rospy.loginfo('Rewards accumulator node is ready.')

    def accumulate_reward_clouds_cb(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Point cloud is received')

        # Discard old messages.
        msg_stamp = rospy.Time.now()
        age = (msg_stamp - pc_msg.header.stamp).to_sec()
        if age > self.max_age:
            rospy.logwarn('Rewards accumulator: Discarding points %.1f s > %.1f s old.', age, self.max_age)
            return

        self.map_frame = pc_msg.header.frame_id

        # try:
        #     transform = self.tf.lookup_transform('world', pc_msg.header.frame_id, rospy.Time(0))
        # except tf2_ros.LookupException:
        #     rospy.logwarn('No transform between estimated robot pose and its ground truth')
        #     return

        t0 = timer()

        # Transform local map to ground truth localization frame
        local_map = numpify(pc_msg)
        local_map = np.vstack([local_map[f] for f in ['x', 'y', 'z', 'reward']])

        local_map = torch.as_tensor(local_map, dtype=torch.float32).transpose(1, 0)[None].to(self.device)
        assert local_map.dim() == 3
        assert local_map.shape[2] == 4  # (1, N, 4)
        if self.global_map is None:
            self.global_map = local_map
        assert self.global_map.dim() == 3
        assert self.global_map.shape[2] == 4  # (1, N, 4)

        # determine new points from local_map based on proximity threshold
        with torch.no_grad():
            map_nn = knn_points(p1=local_map, p2=self.global_map, K=1)
            proximity_mask = map_nn.dists.sqrt() > self.dist_th
        assert proximity_mask.shape[:2] == local_map.shape[:2]
        self.new_points = local_map[:, proximity_mask.squeeze(), :]

        # TODO: transform new points to be on a ground truth mesh, this doesn't work
        # self.new_points = transform_points(self.new_points.squeeze(0).transpose(1, 0), transform).transpose(1, 0).unsqueeze(0)
        rospy.logdebug(f'Points distances, min: {map_nn.dists.sqrt().min()}, mean: {map_nn.dists.sqrt().mean()}')
        rospy.logdebug(f'Adding {self.new_points.shape[1]} new points')
        assert self.new_points.dim() == 3
        assert self.new_points.shape[2] == 4  # (1, n, 4)

        # and accumulate new points to global map
        self.global_map = torch.cat([self.global_map, self.new_points], dim=1)
        assert self.global_map.dim() == 3
        assert self.global_map.shape[2] == 4  # (1, N, 4)
        rospy.logdebug('Point cloud accumulation took: %.3f s', timer() - t0)

    def msgify_reward_cloud(self, cloud):
        assert cloud.shape[1] >= 4
        # publish global map with reward as intensity value
        map = np.zeros(cloud.shape[0], dtype=[('x', np.float32),
                                              ('y', np.float32),
                                              ('z', np.float32),
                                              ('reward', np.float32)])
        map['x'] = cloud[..., 0]
        map['y'] = cloud[..., 1]
        map['z'] = cloud[..., 2]
        map['reward'] = cloud[..., 3]
        # convert point cloud to msg and publish it
        map_msg = msgify(PointCloud2, map)
        map_msg.header.frame_id = self.map_frame
        map_msg.header.stamp = rospy.Time.now()
        return map_msg

    def run(self):
        rate = rospy.Rate(self.reward_cloud_rate)
        while not rospy.is_shutdown():
            if self.global_map is None or self.new_points is None:
                continue
            global_map_msg = self.msgify_reward_cloud(self.global_map.detach().cpu().squeeze(0))
            new_map_msg = self.msgify_reward_cloud(self.new_points.detach().cpu().squeeze(0))
            self.rewards_cloud_pub.publish(global_map_msg)
            self.new_cloud_pub.publish(new_map_msg)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('rewards_accumulator', log_level=rospy.INFO)
    proc = RewardsAccumulator(reward_cloud_topic=rospy.get_param('~reward_cloud_topic', 'reward_cloud'))
    proc.run()
