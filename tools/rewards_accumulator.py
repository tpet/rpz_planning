#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64
from ros_numpy import msgify, numpify
import tf2_ros

import torch
import numpy as np
from pytorch3d.ops.knn import knn_points
from timeit import default_timer as timer


def transform_points(points, T):
    assert isinstance(points, torch.Tensor)
    assert len(points.shape) == 2
    assert points.shape[0] == 3 or points.shape[0] == 4  # (3, N) or (4, N)
    assert isinstance(T, torch.Tensor)
    assert T.shape == (4, 4)
    # Transform local map to ground truth localization frame
    T = T.to(points.device)
    R, t = T[:3, :3], T[:3, 3]
    points[:3, ...] = torch.matmul(R, points[:3, ...]) + t.reshape([3, 1])
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
        device_id = rospy.get_param('~gpu_id', 0)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_id))
            rospy.loginfo("Using GPU device id: %i, name: %s", device_id, torch.cuda.get_device_name(device_id))
        else:
            rospy.loginfo("Using CPU")
            self.device = torch.device("cpu")
        self.global_map = None
        self.local_map = None
        self.new_map = None
        self.map_frame = None

        # any point that is farer than this threshold from points in existing pcl is considered as new
        self.dist_th = rospy.get_param('~pts_proximity_th', 0.5)
        self.max_age = rospy.get_param('~max_age', 10.0)

        self.reward_cloud_rate = rospy.get_param('~reward_cloud_rate', 1.0)
        self.rewards_cloud_pub = rospy.Publisher('~rewards_map', PointCloud2, queue_size=1)
        self.new_cloud_pub = rospy.Publisher('~new_rewards_map', PointCloud2, queue_size=1)
        self.reward_pub = rospy.Publisher('~total_reward', Float64, queue_size=1)

        self.tf = tf2_ros.Buffer(cache_time=rospy.Duration(100))
        self.tl = tf2_ros.TransformListener(self.tf)

        self.local_map_sub = rospy.Subscriber(reward_cloud_topic, PointCloud2, self.accumulate_reward_clouds_cb)
        rospy.loginfo('Rewards accumulator node is ready.')

    def accumulate_reward_clouds_cb(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Point cloud is received')

        # # Discard old messages.
        # msg_stamp = rospy.Time.now()
        # age = (msg_stamp - pc_msg.header.stamp).to_sec()
        # if age > self.max_age:
        #     rospy.logwarn('Rewards accumulator: Discarding points %.1f s > %.1f s old.', age, self.max_age)
        #     return

        self.map_frame = pc_msg.header.frame_id

        try:
            transform1 = self.tf.lookup_transform('X1', pc_msg.header.frame_id,
                                                  pc_msg.header.stamp, rospy.Duration(30))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            rospy.logwarn('Map accumulator: No transform between estimated robot pose and its ground truth')
            return

        try:
            transform2 = self.tf.lookup_transform(pc_msg.header.frame_id, 'X1_ground_truth',
                                                  pc_msg.header.stamp, rospy.Duration(30))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            rospy.logwarn('Map accumulator: No transform between estimated robot pose and its ground truth')
            return

        t0 = timer()

        # Transform local map to ground truth localization frame
        local_map = numpify(pc_msg)
        local_map = np.vstack([local_map[f] for f in ['x', 'y', 'z', 'reward']])

        # transform new points to be on a ground truth mesh
        local_map = torch.as_tensor(local_map, dtype=torch.float32).transpose(1, 0).to(self.device)
        p = local_map.transpose(1, 0)
        T1 = numpify(transform1.transform)
        T2 = numpify(transform2.transform)
        T = torch.as_tensor(np.matmul(T2, T1), dtype=torch.float32).to(p.device)
        local_map = transform_points(p, T).transpose(1, 0)

        assert local_map.dim() == 2
        assert local_map.shape[1] == 4  # (N, 4)
        if self.global_map is None:
            self.global_map = local_map
            self.local_map = local_map
        assert self.global_map.dim() == 2
        assert self.global_map.shape[1] == 4  # (N, 4)

        # determine new points from local_map based on proximity threshold
        with torch.no_grad():
            # map_nn = knn_points(p1=local_map.unsqueeze(0), p2=self.local_map.unsqueeze(0), K=1)
            map_nn = knn_points(p1=local_map.unsqueeze(0), p2=self.global_map.unsqueeze(0), K=1)
            dists, idxs = map_nn.dists.sqrt().squeeze(), map_nn.idx.squeeze()
            proximity_mask = dists > self.dist_th
        assert len(dists) == local_map.shape[0]
        assert len(idxs) == local_map.shape[0]
        self.new_map = local_map[proximity_mask, :]
        self.local_map = local_map

        rospy.logdebug(f'Points distances, min: {map_nn.dists.sqrt().min()}, mean: {map_nn.dists.sqrt().mean()}')
        rospy.logdebug(f'Adding {self.new_map.shape[1]} new points')
        assert self.new_map.dim() == 2
        assert self.new_map.shape[1] == 4  # (n, 4)

        # and accumulate new points to global map
        self.global_map = torch.cat([self.global_map, self.new_map], dim=0)
        # accumulate rewards (max or log odds), compare global and latest local map rewards
        rospy.logdebug(f'Global map probabilities: {torch.min(self.global_map[:, 3])} ~ {torch.max(self.global_map[:, 3])}')
        print(local_map[~proximity_mask, 3].shape)
        rospy.logdebug(f'Common map probabilities {torch.min(local_map[~proximity_mask, 3])} ~ {torch.max(local_map[~proximity_mask, 3])}')
        self.global_map[idxs[~proximity_mask], 3] = torch.max(self.global_map[idxs[~proximity_mask], 3],
                                                              local_map[~proximity_mask, 3])
        assert self.global_map.dim() == 2
        assert self.global_map.shape[1] == 4  # (N, 4)
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
            if self.global_map is None or self.new_map is None:
                continue
            # publish reward clouds
            global_map_msg = self.msgify_reward_cloud(self.global_map.detach().cpu())
            new_map_msg = self.msgify_reward_cloud(self.new_map.detach().cpu())
            self.rewards_cloud_pub.publish(global_map_msg)
            self.new_cloud_pub.publish(new_map_msg)
            # publish total reward value
            total_reward = self.global_map[:, 3].sum().detach().item()
            rospy.loginfo('Total reward: %.1f', total_reward)
            rospy.loginfo('New observations reward: %.1f', self.new_map[:, 3].sum().detach().item())
            self.reward_pub.publish(Float64(total_reward))
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('rewards_accumulator', log_level=rospy.INFO)
    proc = RewardsAccumulator(reward_cloud_topic=rospy.get_param('~reward_cloud_topic', 'reward_cloud'))
    proc.run()
