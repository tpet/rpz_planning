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
        self.normalize = True

        self.load_gt_mesh(path_to_obj)

        self.map_sub = rospy.Subscriber(map_topic, PointCloud2, self.pc_callback)

    def load_gt_mesh(self, path_to_obj):
        verts, faces, aux = load_obj(path_to_obj)

        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        # For this tutorial, normals and textures are ignored.
        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)

        if self.normalize:
            # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
            center = verts.mean(0)
            verts = verts - center
            scale = max(verts.abs().max(0)[0])
            verts = verts / scale

        # We construct a Meshes structure for the target mesh
        self.map_gt_mesh = Meshes(verts=[verts], faces=[faces_idx])

    def pc_callback(self, pc_msg, n_sample_points=5000):
        t0 = timer()
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Received point cloud message')
        map_np = numpify(pc_msg)
        # remove inf points
        mask = np.isfinite(map_np['x']) & np.isfinite(map_np['y']) & np.isfinite(map_np['z'])
        map_np = map_np[mask]
        if n_sample_points is not None:
            if n_sample_points < len(map_np):
                map_np = map_np[np.random.choice(len(map_np), n_sample_points)]
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
        # self.map_sub.unregister()
        # rospy.logwarn('Map topic is unsubscribed.')


if __name__ == '__main__':
    rospy.init_node('map_eval', log_level=rospy.INFO)
    proc = MapEval(map_topic='/X1/map_accumulator/map',
                   path_to_obj=os.path.join(os.path.dirname(__file__), 'world_to_mesh/meshes/simple_tunnel_01_gz.obj'))
    rospy.loginfo('Mapping evaluation node is initialized')
    rospy.spin()
