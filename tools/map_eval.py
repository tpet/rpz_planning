#!/usr/bin/env python

import os
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import point_mesh_edge_distance, point_mesh_face_distance
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
from timeit import default_timer as timer


class MapEval:
    """
    This ROS node is simply subscribes to global map topic with PointCloud2 msgs
    and compares it to ground truth mesh of the simulated environment.
    Metrics for comparison are taken from here:
    https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_edge_distance
    """
    def __init__(self, map_topic, path_to_obj):
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)
        # Set the device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self.device = device
        self.map_gt_mesh = None
        self.normalize = rospy.get_param('~normalize_mesh_pcl', False)
        self.subscribe_once = rospy.get_param('~subscribe_once', False)
        self.do_points_sampling = rospy.get_param('~do_points_sampling', True)
        self.n_sample_points = rospy.get_param('~n_sample_points', 5000)

        t = timer()
        self.load_gt_mesh(path_to_obj)
        rospy.loginfo('Loading ground truth mesh took %.3f s', timer() - t)

        self.map_frame = None
        # TODO: rewrite it as a server-client
        # self.map_sub = rospy.Subscriber(map_topic, PointCloud2, self.pc_callback)
        self.publish_ground_truth(n_points=self.n_sample_points)

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

    def publish_ground_truth(self, n_points=5000, rate=1):
        if self.map_gt_mesh is not None:
            assert isinstance(self.map_gt_mesh, Meshes)
            rate = rospy.Rate(rate)
            while not rospy.is_shutdown():
                sampled_points = sample_points_from_meshes(self.map_gt_mesh, n_points)
                points_to_pub = sampled_points.squeeze().cpu().numpy().transpose(1, 0)
                map_frame = self.map_frame if self.map_frame is not None else 'subt'
                self.publish_pointcloud(points_to_pub,
                                        topic_name='~cloud_from_gt_mesh',
                                        stamp=rospy.Time.now(),
                                        frame_id=map_frame)
                rospy.logdebug(f'Ground truth mesh frame: {map_frame}')
                rospy.loginfo(f'Publishing points of shape {points_to_pub.shape} sampled from ground truth mesh')
                rate.sleep()

    def load_gt_mesh(self, path_to_obj):
        verts, faces, aux = load_obj(path_to_obj)

        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        # For this tutorial, normals and textures are ignored.
        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)

        # TODO: do normalization only for losses computation
        if self.normalize:
            # We scale normalize and center the target mesh
            # to fit in a sphere of radius 1 centered at (0,0,0).
            center = verts.mean(0)
            verts = verts - center
            scale = max(verts.abs().max(0)[0])
            verts = verts / scale

        # We construct a Meshes structure for the target mesh
        self.map_gt_mesh = Meshes(verts=[verts], faces=[faces_idx])

    def pc_callback(self, pc_msg):
        t0 = timer()
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Received point cloud message')
        self.map_frame = pc_msg.header.frame_id
        map_np = numpify(pc_msg)
        # remove inf points
        mask = np.isfinite(map_np['x']) & np.isfinite(map_np['y']) & np.isfinite(map_np['z'])
        map_np = map_np[mask]
        if self.do_points_sampling:
            if self.n_sample_points < len(map_np):
                map_np = map_np[np.random.choice(len(map_np), self.n_sample_points)]
        assert len(map_np.shape) == 1
        # pull out x, y, and z values
        map = torch.zeros([1] + list(map_np.shape) + [3], dtype=torch.float32)
        map[..., 0] = torch.tensor(map_np['x'])
        map[..., 1] = torch.tensor(map_np['y'])
        map[..., 2] = torch.tensor(map_np['z'])
        if self.normalize:
            # We scale normalize and center the point cloud to fit in a sphere of radius 1 centered at (0,0,0).
            center = map.mean(1)
            map = map - center
            scale = max(map.abs().max(1)[0])
            map = map / scale
        assert map.dim() == 3
        assert map.size()[2] == 3
        point_cloud = Pointclouds(map).to(self.device)
        # compare point cloud to mesh here
        t1 = timer()
        loss_edge = point_mesh_edge_distance(meshes=self.map_gt_mesh, pcls=point_cloud)
        loss_face = point_mesh_face_distance(meshes=self.map_gt_mesh, pcls=point_cloud)
        rospy.loginfo(f'Loss Edge: {loss_edge.detach().cpu().numpy():.6f}, \
                        Loss Face: {loss_face.detach().cpu().numpy():.6f}')
        rospy.logdebug('Point cloud conversion took: %.3f s', t1 - t0)
        rospy.loginfo('Mapping evaluation took: %.3f s', timer()-t1)

        if self.subscribe_once:
            self.map_sub.unregister()
            rospy.logwarn('Map topic is unsubscribed.')


if __name__ == '__main__':
    rospy.init_node('map_eval', log_level=rospy.DEBUG)
    path_fo_gt_map_mesh = rospy.get_param('~gt_mesh')
    assert os.path.exists(path_fo_gt_map_mesh)
    rospy.loginfo('Using ground truth mesh file: %s', path_fo_gt_map_mesh)
    proc = MapEval(map_topic=rospy.get_param('~map_frame', 'map_accumulator/map'),
                   path_to_obj=path_fo_gt_map_mesh)
    rospy.loginfo('Mapping evaluation node is initialized')
    rospy.spin()
