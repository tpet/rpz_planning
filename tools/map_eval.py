#!/usr/bin/env python

import os
import torch
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from rpz_planning import point_face_distance_truncated
from rpz_planning import point_edge_distance_truncated
from rpz_planning import face_point_distance_truncated
from rpz_planning import edge_point_distance_truncated
from rpz_planning import chamfer_distance_truncated
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
from timeit import default_timer as timer
import xlwt
# problems with Cryptodome: pip install pycryptodomex
# https://github.com/DP-3T/reference_implementation/issues/1


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
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.map_gt_mesh = None
        self.map_gt = None
        self.normalize = rospy.get_param('~normalize_mesh_pcl', False)
        self.do_points_sampling = rospy.get_param('~do_points_sampling', False)
        self.do_eval = rospy.get_param('~do_eval', True)
        self.load_gt = rospy.get_param('~load_gt', True)
        self.record_metrics = rospy.get_param('~record_metrics', True)
        self.xls_file = rospy.get_param('~metrics_xls_file', f'/tmp/mapping-eval_{timer()}.xls')
        self.n_sample_points = rospy.get_param('~n_sample_points', 5000)
        self.max_age = rospy.get_param('~max_age', 1.0)
        self.rate = rospy.get_param('~eval_rate', 1.0)
        # exploration losses publishers
        self.face_loss_pub = rospy.Publisher('~exp_loss_face', Float64, queue_size=1)
        self.edge_loss_pub = rospy.Publisher('~exp_loss_edge', Float64, queue_size=1)
        self.chamfer_loss_pub = rospy.Publisher('~exp_loss_chamfer', Float64, queue_size=1)
        # mapping accuracy publishers
        self.map_face_acc_pub = rospy.Publisher('~map_loss_face', Float64, queue_size=1)
        self.map_edge_acc_pub = rospy.Publisher('~map_loss_edge', Float64, queue_size=1)
        self.map_chamfer_acc_pub = rospy.Publisher('~map_loss_chamfer', Float64, queue_size=1)

        t = timer()
        if self.load_gt:
            self.load_gt_mesh(path_to_obj)
            rospy.loginfo('Loading ground truth mesh took %.3f s', timer() - t)
        rospy.loginfo('Mapping evaluation node is ready')

        # record and publish the metrics
        if self.record_metrics:
            self.wb = xlwt.Workbook()
            self.ws_writer = self.wb.add_sheet(f"Exp_{timer()}".replace('.', '_'))
            self.ws_writer.write(0, 0, 'Exploration Face loss')
            self.ws_writer.write(0, 1, 'Exploration Edge loss')
            self.ws_writer.write(0, 2, 'Exploration Chamfer loss')
            self.ws_writer.write(0, 3, 'Map Face loss')
            self.ws_writer.write(0, 4, 'Map Edge loss')
            self.ws_writer.write(0, 5, 'Map Chamfer loss')
            self.ws_writer.write(0, 6, 'Time stamp')
            self.row_number = 1

        self.map_frame = rospy.get_param('~map_frame', 'subt')
        self.pc_msg = None
        # TODO: rewrite it as a server-client
        rospy.loginfo('Subscribing to map topic: %s', map_topic)
        self.map_sub = rospy.Subscriber(map_topic, PointCloud2, self.pc_callback)
        self.run()


    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id):
        assert isinstance(points, np.ndarray)
        assert len(points.shape) == 2
        assert points.shape[0] == 3
        # create PointCloud2 msg
        data = np.zeros(points.shape[1], dtype=[('x', np.float32),
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

    def run(self):
        if self.map_gt_mesh is not None:
            assert isinstance(self.map_gt_mesh, Meshes)
            rate = rospy.Rate(self.rate)
            while not rospy.is_shutdown():
                # compare subscribed map point cloud to ground truth mesh
                if self.do_eval:
                    self.eval()

                # publish point cloud from ground truth mesh
                if self.map_gt is not None:
                    points_to_pub = self.map_gt.squeeze().cpu().numpy().transpose(1, 0)

                    # selecting default map frame if it is not available in ROS
                    map_frame = self.map_frame if self.map_frame is not None else 'subt'
                    self.publish_pointcloud(points_to_pub,
                                            topic_name='~cloud_from_gt_mesh',
                                            stamp=rospy.Time.now(),
                                            frame_id=map_frame)
                    rospy.logdebug(f'Ground truth mesh frame: {map_frame}')
                    rospy.logdebug(f'Publishing points of shape {points_to_pub.shape} sampled from ground truth mesh')
                rate.sleep()

    def load_gt_mesh(self, path_to_mesh_file):
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
        gt_mesh_faces_idx = gt_mesh_faces_idx.to(self.device)
        gt_mesh_verts = gt_mesh_verts.to(self.device)
        # TODO: correct coordinates mismatch in Blender (swap here X and Y)
        R = torch.tensor([[ 0., 1., 0.],
                          [-1., 0., 0.],
                          [ 0., 0., 1.]]).to(self.device)
        gt_mesh_verts = torch.matmul(R, gt_mesh_verts.transpose(1, 0)).transpose(1, 0)
        assert gt_mesh_verts.shape[1] == 3

        # We construct a Meshes structure for the target mesh
        self.map_gt_mesh = Meshes(verts=[gt_mesh_verts], faces=[gt_mesh_faces_idx]).to(self.device)
        if self.do_points_sampling:
            self.map_gt = sample_points_from_meshes(self.map_gt_mesh, self.n_sample_points).to(self.device)
        else:
            self.map_gt = self.map_gt_mesh.verts_packed().to(self.device)
        rospy.logdebug(f'Loaded mesh with verts shape: {gt_mesh_verts.size()}')

    def pc_callback(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Received point cloud message')
        self.pc_msg = pc_msg

    def eval(self):
        if self.pc_msg is None:
            rospy.logwarn('No points received')
            return
        assert isinstance(self.pc_msg, PointCloud2)
        # Discard old messages.
        msg_stamp = rospy.Time.now()
        age = (msg_stamp - self.pc_msg.header.stamp).to_sec()
        if age > self.max_age:
            rospy.logwarn('Discarding points %.1f s > %.1f s old.', age, self.max_age)
            return
        t0 = timer()
        map_np = numpify(self.pc_msg)
        # remove inf points
        mask = np.isfinite(map_np['x']) & np.isfinite(map_np['y']) & np.isfinite(map_np['z'])
        map_np = map_np[mask]
        if self.do_points_sampling:
            if self.n_sample_points < len(map_np):
                map_np = map_np[np.random.choice(len(map_np), self.n_sample_points)]
        assert len(map_np.shape) == 1
        # pull out x, y, and z values
        map = torch.zeros([1] + list(map_np.shape) + [3], dtype=torch.float32).to(self.device)
        map[..., 0] = torch.tensor(map_np['x'])
        map[..., 1] = torch.tensor(map_np['y'])
        map[..., 2] = torch.tensor(map_np['z'])
        assert map.dim() == 3
        assert map.size()[2] == 3
        rospy.logdebug('Point cloud conversion took: %.3f s', timer() - t0)

        # calculate mapping metrics
        assert map.dim() == 3
        assert map.size()[2] == 3
        point_cloud = Pointclouds(map).to(self.device)
        # compare point cloud to mesh here
        with torch.no_grad():
            t1 = timer()
            exp_loss_edge = edge_point_distance_truncated(meshes=self.map_gt_mesh, pcls=point_cloud)
            # distance between mesh and points is computed as a distance from point to triangle
            # if point's projection is inside triangle, then the distance is computed as along
            # a normal to triangular plane. Otherwise as a distance to closest edge of the triangle:
            # https://github.com/facebookresearch/pytorch3d/blob/fe39cc7b806afeabe64593e154bfee7b4153f76f/pytorch3d/csrc/utils/geometry_utils.h#L635
            exp_loss_face = face_point_distance_truncated(meshes=self.map_gt_mesh, pcls=point_cloud)
            exp_loss_chamfer = chamfer_distance_truncated(x=self.map_gt, y=map)
            t2 = timer()
            rospy.loginfo('Explored space evaluation took: %.3f s\n', t2 - t1)

            # `exp_loss_face`, `exp_loss_edge` and `exp_loss_chamfer` describe exploration progress
            # current map accuracy could be evaluated by computing vice versa distances:
            # - from points in cloud to mesh faces/edges:
            map_loss_edge = point_edge_distance_truncated(meshes=self.map_gt_mesh, pcls=point_cloud)
            map_loss_face = point_face_distance_truncated(meshes=self.map_gt_mesh, pcls=point_cloud)
            # - from points in cloud to nearest neighbours of points sampled from mesh:
            map_loss_chamfer = chamfer_distance_truncated(x=map, y=self.map_gt,
                                                          apply_point_reduction=True,
                                                          batch_reduction='mean', point_reduction='mean')
            rospy.loginfo('Mapping accuracy evaluation took: %.3f s\n', timer() - t2)

        # record data
        if self.record_metrics:
            self.ws_writer.write(self.row_number, 0, f'{exp_loss_face.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 1, f'{exp_loss_edge.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 2, f'{exp_loss_chamfer.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 3, f'{map_loss_face.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 4, f'{map_loss_edge.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 5, f'{map_loss_chamfer.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 6, msg_stamp.to_sec())
            self.row_number += 1
            self.wb.save(self.xls_file)

        print('\n')
        rospy.loginfo(f'Exploration Edge loss: {exp_loss_edge.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration Face loss: {exp_loss_face.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration Chamfer loss: {exp_loss_chamfer.squeeze().detach().cpu().numpy():.3f}')
        print('-'*30)

        print('\n')
        rospy.loginfo(f'Map Edge loss: {map_loss_edge.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Map Face loss: {map_loss_face.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Map Chamfer loss: {map_loss_chamfer.squeeze().detach().cpu().numpy():.3f}')
        print('-' * 30)

        # publish losses and metrics
        self.face_loss_pub.publish(Float64(exp_loss_face.detach().cpu().numpy()))
        self.edge_loss_pub.publish(Float64(exp_loss_edge.detach().cpu().numpy()))
        self.chamfer_loss_pub.publish(Float64(exp_loss_chamfer.detach().cpu().numpy()))

        self.map_face_acc_pub.publish(Float64(map_loss_face.detach().cpu().numpy()))
        self.map_edge_acc_pub.publish(Float64(map_loss_edge.detach().cpu().numpy()))
        self.map_chamfer_acc_pub.publish(Float64(map_loss_chamfer.squeeze().detach().cpu().numpy()))


if __name__ == '__main__':
    rospy.init_node('map_eval', log_level=rospy.INFO)
    path_fo_gt_map_mesh = rospy.get_param('~gt_mesh')
    assert os.path.exists(path_fo_gt_map_mesh)
    rospy.loginfo('Using ground truth mesh file: %s', path_fo_gt_map_mesh)
    proc = MapEval(map_topic=rospy.get_param('~map_topic', 'map_accumulator/map'),
                   path_to_obj=path_fo_gt_map_mesh)
    rospy.loginfo('Mapping evaluation node is initialized')
    rospy.spin()
