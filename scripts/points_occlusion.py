#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo
import tf
import tf2_ros
import torch
import time
import cv2
import numpy as np
from rpz_planning import point_visibility
from ros_numpy import msgify, numpify
from pyquaternion import Quaternion


class PointsProcessor:
    def __init__(self,
                 pc_topic='/final_cost_cloud',
                 cam_info_topics=['/viz/camera_0/camera_info',
                                  # '/viz/camera_1/camera_info',
                                  # '/viz/camera_2/camera_info',
                                  # '/viz/camera_3/camera_info',
                                  # '/viz/camera_4/camera_info',
                                  # '/viz/camera_5/camera_info'
                                  ],
                 min_dist=1.0,
                 max_dist=15.0,
                 ):
        self.pc_frame = None
        self.points = None
        self.pc_clip_limits = [rospy.get_param('~frustum_min_dist', min_dist),
                               rospy.get_param('~frustum_max_dist', max_dist)]
        if torch.cuda.is_available():
            self.device = torch.device("cpu")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.pc_topic = rospy.get_param('~pointcloud_topic', pc_topic)
        print("Subscribed to " + self.pc_topic)
        pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pc_callback)

        for cam_info_topic in cam_info_topics:
            print("Subscribed to " + cam_info_topic)
            cam_info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self.cam_info_callback)

        self.tl = tf.TransformListener()

    @staticmethod
    def ego_to_cam_torch(points, trans, quat):
        """Transform points (3 x N) from ego frame into a pinhole camera
        """
        assert isinstance(points, torch.Tensor)
        assert isinstance(trans, torch.Tensor)
        assert isinstance(quat, Quaternion)
        assert points.dim() == 2
        assert points.shape[0] == 3
        assert trans.shape == (3, 1)
        points = points - trans
        R_inv = torch.tensor(quat.normalised.inverse.rotation_matrix, dtype=torch.float32)
        points = R_inv.matmul(points)
        return points

    def get_cam_frustum_pts(self, points, img_height, img_width, intrins):
        assert isinstance(points, torch.Tensor)
        assert isinstance(intrins, torch.Tensor)
        assert points.dim() == 2
        assert points.shape[0] == 3
        # clip points between MIN_DIST and MAX_DIST meters distance from the camera
        dist_mask = (points[2] > self.pc_clip_limits[0]) & (points[2] < self.pc_clip_limits[1])

        # find points that are observed by the camera (in its FOV)
        pts_homo = intrins[:3, :3].matmul(points)
        pts_homo[:2] /= pts_homo[2:3]
        fov_mask = (pts_homo[2] > 0) & \
                   (pts_homo[0] > 1) & (pts_homo[0] < img_width - 1) & \
                   (pts_homo[1] > 1) & (pts_homo[1] < img_height - 1)
        points = points[:, dist_mask & fov_mask]
        return points

    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id):
        assert isinstance(points, np.ndarray)
        assert len(points.shape) == 2
        assert points.shape[0] == 3
        # create PointCloud2 msg
        data = np.zeros(points.shape[1],
                        dtype=[('x', np.float32),
                               ('y', np.float32),
                               ('z', np.float32)])
        data['x'] = points[0, ...]
        data['y'] = points[1, ...]
        data['z'] = points[2, ...]
        pc_msg = msgify(PointCloud2, data)
        pc_msg.header.stamp = stamp
        pc_msg.header.frame_id = frame_id
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def pc_callback(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        points = numpify(pc_msg)
        # remove inf points
        mask = np.isfinite(points['x']) & np.isfinite(points['y']) & np.isfinite(points['z'])
        points = points[mask]
        # pull out x, y, and z values
        self.points = torch.zeros([3] + list(points.shape), dtype=torch.float32)
        self.points[0, ...] = torch.tensor(points['x'])
        self.points[1, ...] = torch.tensor(points['y'])
        self.points[2, ...] = torch.tensor(points['z'])
        self.pc_frame = pc_msg.header.frame_id

    def cam_info_callback(self, cam_info_msg):
        t0 = time.time()
        fovH = cam_info_msg.height
        fovW = cam_info_msg.width

        cam_frame = cam_info_msg.header.frame_id
        K = torch.zeros((4, 4))
        K[0][0] = cam_info_msg.K[0]
        K[0][2] = cam_info_msg.K[2]
        K[1][1] = cam_info_msg.K[4]
        K[1][2] = cam_info_msg.K[5]
        K[2][2] = 1.
        K[3][3] = 1.
        K = K.float().to(self.device)

        if self.pc_frame is not None:  # and self.tl.frameExists(self.pc_frame):
            self.run(fovH, fovW, K, cam_frame, output_pc_topic='/output/pointcloud')
        # print(f'[INFO]: Callback run time {1000 * (time.time() - t0):.1f} ms')

    def run(self, img_height, img_width, intrins, cam_frame, output_pc_topic='/output/pointcloud'):
        t1 = time.time()
        # find transformation between lidar and camera
        t = self.tl.getLatestCommonTime(self.pc_frame, cam_frame)
        trans, quat = self.tl.lookupTransform(self.pc_frame, cam_frame, t)

        # transform point cloud to camera frame
        quat = Quaternion(w=quat[3], x=quat[0], y=quat[1], z=quat[2])
        points = self.points
        trans = torch.tensor(trans, dtype=torch.float32).view(3, 1)
        points = self.ego_to_cam_torch(points, trans, quat)
        assert points.shape[0] == 3

        # clip points between MIN_DIST and MAX_DIST meters distance from the camera
        points = self.get_cam_frustum_pts(points, img_height, img_width, intrins)
        assert points.shape[0] == 3

        self.publish_pointcloud(points.numpy(),
                                output_pc_topic,
                                rospy.Time.now(),
                                cam_frame)
        # remove hidden points from current camera FOV
        vis_mask = point_visibility(points.transpose(1, 0), origin=torch.zeros([1, 3]))
        points = points[:, torch.tensor(vis_mask.detach(), dtype=torch.bool)]
        assert points.shape[0] == 3
        rospy.loginfo('Number of observed points from %s is: %i', cam_frame, points.shape[1])

        self.publish_pointcloud(points.numpy(),
                                output_pc_topic+'_visible',
                                rospy.Time.now(),
                                cam_frame)
        rospy.loginfo('Processing took %f ms', 1000*(time.time()-t1))


if __name__ == '__main__':
    rospy.init_node('pc_processor_node')
    proc = PointsProcessor()
    rospy.spin()
