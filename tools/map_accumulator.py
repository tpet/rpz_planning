#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
from pyquaternion import Quaternion
from timeit import default_timer as timer
import yaml
from rosbag.bag import Bag
import torch
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from pcl_mesh_metrics import face_point_distance_weighted
from pcl_mesh_metrics import edge_point_distance_weighted
import os


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
        self.bagfile = rospy.get_param('~bagfile', None)
        self.msg_counter = 0
        if self.bagfile is not None:
            self.bag_info = self.get_bag_info(self.bagfile)
        self.do_eval = rospy.get_param('~do_eval', None)
        self.path_to_gt_mesh = rospy.get_param('~gt_mesh', None)
        rospy.Subscriber('local_map', PointCloud2, self.pc_callback)
        rospy.loginfo('Map accumulator node is ready. You may need to heat Space for bagfile to start.')

    @staticmethod
    def load_ground_truth(path_to_mesh_file, n_sample_points=5000, device=torch.device("cuda:0")):
        assert os.path.exists(path_to_mesh_file)
        if '.obj' in path_to_mesh_file:
            gt_mesh_verts, faces, _ = load_obj(path_to_mesh_file)
            gt_mesh_faces_idx = faces.verts_idx
        elif '.ply' in path_to_mesh_file:
            gt_mesh_verts, gt_mesh_faces_idx = load_ply(path_to_mesh_file)
        else:
            rospy.logerr('Supported mesh formats are *.obj or *.ply')
            return
        # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
        # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        # For this tutorial, normals and textures are ignored.
        gt_mesh_faces_idx = gt_mesh_faces_idx.to(device)
        gt_mesh_verts = gt_mesh_verts.to(device)
        # TODO: correct coordinates mismatch in Blender (swap here X and Y)
        R = torch.tensor([[ 0., 1., 0.],
                          [-1., 0., 0.],
                          [ 0., 0., 1.]]).to(device)
        gt_mesh_verts = torch.matmul(R, gt_mesh_verts.transpose(1, 0)).transpose(1, 0)
        assert gt_mesh_verts.shape[1] == 3

        # We construct a Meshes structure for the target mesh
        map_gt_mesh = Meshes(verts=[gt_mesh_verts], faces=[gt_mesh_faces_idx]).to(device)
        map_gt_pcl = sample_points_from_meshes(map_gt_mesh, n_sample_points).to(device)
        rospy.logdebug(f'Loaded mesh with verts shape: {gt_mesh_verts.size()}')
        return map_gt_mesh, map_gt_pcl

    @staticmethod
    def get_bag_info(bagfile):
        info_dict = yaml.load(Bag(bagfile, 'r')._get_yaml_info(), Loader=yaml.FullLoader)
        return info_dict

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
        self.msg_counter += 1
        # Transform the point cloud
        self.map_frame = pc_msg.header.frame_id
        try:
            t0 = timer()
            transform = self.tf.lookup_transform('X1_ground_truth', 'X1', rospy.Time(0))
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

            if self.bagfile is not None:
                t_remains = self.bag_info['end'] - rospy.Time.now().to_sec()
                if t_remains < 0.5 and self.do_eval:  # [sec]
                    # save resultant map here or compare it to ground truth mesh
                    # and output metrics
                    rospy.loginfo('Loading ground truth mesh...')
                    mesh_gt, pcl_gt = self.load_ground_truth(self.path_to_gt_mesh, device=self.device)
                    map = torch.as_tensor(points3[None], dtype=torch.float32).transpose(2, 1).to(self.device)
                    self.eval(map, map_gt_mesh=mesh_gt, map_gt_pcl=pcl_gt)
                    self.do_eval = False
        except tf2_ros.LookupException:
            rospy.logwarn('No transform between estimated robot pose and its ground truth')

    def eval(self, map, map_gt_mesh, map_gt_pcl):
        assert isinstance(map, torch.Tensor)
        assert map.dim() == 3
        assert map.size()[2] == 3
        point_cloud = Pointclouds(map).to(self.device)
        # compare point cloud to mesh here
        with torch.no_grad():
            loss_edge = edge_point_distance_weighted(meshes=map_gt_mesh, pcls=point_cloud)
            # distance between mesh and points is computed as a distance from point to triangle
            # if point's projection is inside triangle, then the distance is computed as along
            # a normal to triangular plane. Otherwise as a distance to closest edge of the triangle:
            # https://github.com/facebookresearch/pytorch3d/blob/fe39cc7b806afeabe64593e154bfee7b4153f76f/pytorch3d/csrc/utils/geometry_utils.h#L635
            loss_face = face_point_distance_weighted(meshes=map_gt_mesh, pcls=point_cloud)
            loss_icp, _ = chamfer_distance(map, map_gt_pcl)

            rospy.loginfo('\n')
            rospy.loginfo(f'Edge loss: {loss_edge.detach().cpu().numpy():.3f}')
            rospy.loginfo(f'Face loss: {loss_face.detach().cpu().numpy():.3f}')
            rospy.loginfo(f'Chamfer loss: {loss_icp.detach().cpu().numpy():.3f}')


if __name__ == '__main__':
    rospy.init_node('map_accumulator', log_level=rospy.INFO)
    proc = MapAccumulator()
    rospy.spin()
