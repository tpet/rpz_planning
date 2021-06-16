#!/usr/bin/env python

import os
import time

import torch
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.knn import knn_points
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
from visualization_msgs.msg import Marker
import rospkg
from scipy.spatial.transform import Rotation
# problems with Cryptodome: pip install pycryptodomex
# https://github.com/DP-3T/reference_implementation/issues/1


class MapEval:
    """
    This ROS node subscribes to global map topic with PointCloud2 msgs
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
        # parameters
        self.normalize = rospy.get_param('~normalize_mesh_pcl', False)
        self.do_points_sampling = rospy.get_param('~do_points_sampling', False)
        self.do_eval = rospy.get_param('~do_eval', True)
        self.load_gt = rospy.get_param('~load_gt', True)
        self.record_metrics = rospy.get_param('~record_metrics', True)
        self.xls_file = rospy.get_param('~metrics_xls_file', f'/tmp/mapping-eval')
        self.xls_file = f'{self.xls_file}_{timer()}.xls'
        self.n_sample_points = rospy.get_param('~n_sample_points', 10000)
        self.coverage_dist_th = rospy.get_param('~coverage_dist_th', 1.0)
        self.max_age = rospy.get_param('~max_age', 1.0)
        self.rate = rospy.get_param('~eval_rate', 1.0)
        # exploration metrics publishers
        self.face_loss_pub = rospy.Publisher('~exp_loss_face', Float64, queue_size=1)
        self.edge_loss_pub = rospy.Publisher('~exp_loss_edge', Float64, queue_size=1)
        self.chamfer_loss_pub = rospy.Publisher('~exp_loss_chamfer', Float64, queue_size=1)
        self.exp_compl_pub = rospy.Publisher('~exp_completeness', Float64, queue_size=1)
        # mapping accuracy publishers
        self.map_face_acc_pub = rospy.Publisher('~map_loss_face', Float64, queue_size=1)
        self.map_edge_acc_pub = rospy.Publisher('~map_loss_edge', Float64, queue_size=1)
        self.map_chamfer_acc_pub = rospy.Publisher('~map_loss_chamfer', Float64, queue_size=1)
        # world ground truth publisher
        self.world_mesh_pub = rospy.Publisher('/world_mesh', Marker, queue_size=1)

        self.map_gt_frame = rospy.get_param('~map_gt_frame', 'subt')
        self.map_gt_mesh = None
        self.map_gt_mesh_marker = Marker()
        self.artifacts_cloud = None
        self.map_gt = None
        self.pc_msg = None
        self.map_frame = None
        self.map = None
        self.coverage_mask = None

        # loading ground truth data
        t = timer()
        if self.load_gt:
            self.load_ground_truth(path_to_obj)
            rospy.loginfo('Loading ground truth took %.3f s', timer() - t)

        # record the metrics
        if self.record_metrics:
            self.wb = xlwt.Workbook()
            self.ws_writer = self.wb.add_sheet(f"Exp_{timer()}".replace('.', '_'))
            self.ws_writer.write(0, 0, 'Time stamp')
            self.ws_writer.write(0, 1, 'Exploration Face loss')
            self.ws_writer.write(0, 2, 'Exploration Edge loss')
            self.ws_writer.write(0, 3, 'Exploration Chamfer loss')
            self.ws_writer.write(0, 4, 'Exploration completeness')
            self.ws_writer.write(0, 5, 'Map Face loss')
            self.ws_writer.write(0, 6, 'Map Edge loss')
            self.ws_writer.write(0, 7, 'Map Chamfer loss')
            self.row_number = 1

        # running the evaluation
        rospy.loginfo('Subscribing to map topic: %s', map_topic)
        self.map_sub = rospy.Subscriber(map_topic, PointCloud2, self.pc_callback)

    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id):
        assert isinstance(points, np.ndarray)
        assert len(points.shape) == 2
        assert points.shape[0] >= 3
        # create PointCloud2 msg
        data = np.zeros(points.shape[1], dtype=[('x', np.float32),
                                                ('y', np.float32),
                                                ('z', np.float32),
                                                ('coverage', np.float32)])
        data['x'] = points[0, ...]
        data['y'] = points[1, ...]
        data['z'] = points[2, ...]
        if points.shape[0] > 3:
            data['coverage'] = points[3, ...]
        pc_msg = msgify(PointCloud2, data)
        pc_msg.header.stamp = stamp
        pc_msg.header.frame_id = frame_id
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def load_ground_truth(self, path_to_mesh_file):
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
        rospy.loginfo(f'Loaded mesh with verts shape: {gt_mesh_verts.size()}')

        # visualization Marker of gt mesh
        marker = Marker()
        marker.header.frame_id = self.map_gt_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "artifacts_ns"
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        r = np.array([[ 0., 0., 1.],
                      [-1., 0., 0.],
                      [ 0., -1., 0.]])
        q = Rotation.from_matrix(r).as_quat()
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 0.2
        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 0
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = f"package://rpz_planning/data/meshes/{self.map_gt_frame}.dae"
        self.map_gt_mesh_marker = marker
        # get the artifacts point cloud
        self.artifacts_cloud = self.get_artifacts_cloud()

    def get_artifacts_cloud(self, artifacts=None):
        time.sleep(2)  # let the tf node load
        # TODO: get artifacts list from tf tree
        if artifacts is None:
            artifacts = ['backpack_1', 'backpack_2', 'backpack_3', 'backpack_4',
                         'phone_1', 'phone_2', 'phone_3', 'phone_4',
                         'rescue_randy_1', 'rescue_randy_2', 'rescue_randy_3', 'rescue_randy_4']
        artifacts_cloud = []
        for i, artifact_name in enumerate(artifacts):
            try:
                artifact_frame = artifact_name
                transform = self.tf.lookup_transform(self.map_gt_frame, artifact_frame, rospy.Time(0))
            except tf2_ros.LookupException:
                rospy.logwarn('Ground truth artifacts poses are not yet available')
                return

            # create artifacts point cloud here from their meshes
            verts, faces, _ = load_obj(os.path.join(rospkg.RosPack().get_path('rpz_planning'),
                                                    f"data/meshes/artifacts/{artifact_name[:-2]}.obj"))
            # mesh = Meshes(verts=[verts], faces=[faces.verts_idx]).to(self.device)
            # cloud = sample_points_from_meshes(mesh, 1000).squeeze(0).to(self.device)
            cloud = verts.cpu().numpy().transpose(1, 0)
            # TODO: correct coordinates mismatch in Blender (swap here X and Y)
            R = np.array([[0., 1., 0.],
                          [-1., 0., 0.],
                          [0., 0., 1.]])
            cloud = np.matmul(R, cloud)
            # transform point cloud to global world frame
            T = numpify(transform.transform)
            R, t = T[:3, :3], T[:3, 3]
            cloud = np.matmul(R, cloud) + t.reshape([3, 1])
            artifacts_cloud.append(cloud)
        artifacts_cloud_np = artifacts_cloud[0]
        for cloud in artifacts_cloud[1:]:
            artifacts_cloud_np = np.concatenate([artifacts_cloud_np, cloud], axis=1)
        assert artifacts_cloud_np.shape[0] == 3
        return artifacts_cloud_np

    def pc_callback(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Received point cloud message')
        self.pc_msg = pc_msg
        self.map_frame = pc_msg.header.frame_id
        try:
            t0 = timer()
            transform = self.tf.lookup_transform(self.map_gt_frame, self.map_frame, rospy.Time(0))
            map_np = numpify(self.pc_msg)
            # remove inf points
            mask = np.isfinite(map_np['x']) & np.isfinite(map_np['y']) & np.isfinite(map_np['z'])
            map_np = map_np[mask]
            assert len(map_np.shape) == 1
            # pull out x, y, and z values
            map = np.zeros([3] + list(map_np.shape), dtype=np.float32)
            map[0, ...] = map_np['x']
            map[1, ...] = map_np['y']
            map[2, ...] = map_np['z']
            T = numpify(transform.transform)
            R, t = T[:3, :3], T[:3, 3]
            map = np.matmul(R, map) + t.reshape([3, 1])

            self.map = torch.as_tensor(map.transpose(1, 0), dtype=torch.float32).unsqueeze(0).to(self.device)
            assert self.map.dim() == 3
            assert self.map.size()[2] == 3
            rospy.logdebug('Point cloud conversion took: %.3f s', timer() - t0)
        except tf2_ros.LookupException:
            rospy.logwarn('No transform between constructed map frame and its ground truth')

    def eval(self, map, map_gt, map_gt_mesh, coverage_dist_th=1.0):
        # Discard old messages.
        msg_stamp = rospy.Time.now()
        age = (msg_stamp - self.pc_msg.header.stamp).to_sec()
        if age > self.max_age:
            rospy.logwarn('Discarding points %.1f s > %.1f s old.', age, self.max_age)
            return None
        assert isinstance(map, torch.Tensor)
        assert map.dim() == 3
        assert map.size()[2] == 3  # (1, N, 3)
        # compare point cloud to mesh here
        with torch.no_grad():
            t1 = timer()
            map_sampled = map.clone()
            if self.do_points_sampling:
                if self.n_sample_points < map.shape[1]:
                    map_sampled = map[:, torch.randint(map.shape[1], (self.n_sample_points,)), :]
            point_cloud = Pointclouds(map_sampled).to(self.device)

            exp_loss_edge = edge_point_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            # distance between mesh and points is computed as a distance from point to triangle
            # if point's projection is inside triangle, then the distance is computed along
            # a normal to triangular plane. Otherwise as a distance to closest edge of the triangle:
            # https://github.com/facebookresearch/pytorch3d/blob/fe39cc7b806afeabe64593e154bfee7b4153f76f/pytorch3d/csrc/utils/geometry_utils.h#L635
            exp_loss_face = face_point_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            exp_loss_chamfer = chamfer_distance_truncated(x=map_gt, y=map)

            # coverage metric (exploration completeness)
            # The metric is evaluated as the fraction of ground truth points considered as covered.
            # A ground truth point is considered as covered if the nearest neighbour
            map_nn = knn_points(p1=map_gt, p2=map,
                                lengths1=torch.tensor(map_gt.shape[1])[None].to(self.device),
                                lengths2=torch.tensor(map.shape[1])[None].to(self.device), K=1)
            coverage_mask = torch.zeros_like(map_nn.dists).to(self.device)
            coverage_mask[map_nn.dists.sqrt() < coverage_dist_th] = 1
            # print(map_nn.dists.sqrt().min(), map_nn.dists.sqrt().mean(), map_nn.dists.sqrt().max())
            assert coverage_mask.shape[:2] == map_gt.shape[:2]
            exp_completeness = coverage_mask.sum() / map_gt.size()[1]
            t2 = timer()

            rospy.logdebug('Explored space evaluation took: %.3f s\n', t2 - t1)

            # `exp_loss_face`, `exp_loss_edge` and `exp_loss_chamfer` describe exploration progress
            # current map accuracy could be evaluated by computing vice versa distances:
            # - from points in cloud to mesh faces/edges:
            map_loss_edge = point_edge_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            map_loss_face = point_face_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            # - from points in cloud to nearest neighbours of points sampled from mesh:
            map_loss_chamfer = chamfer_distance_truncated(x=map, y=map_gt,
                                                          apply_point_reduction=True,
                                                          batch_reduction='mean', point_reduction='mean')
            rospy.logdebug('Mapping accuracy evaluation took: %.3f s\n', timer() - t2)
            rospy.loginfo('Evaluation took: %.3f s\n', timer() - t1)

        # record data
        if self.record_metrics:
            self.ws_writer.write(self.row_number, 0, msg_stamp.to_sec())
            self.ws_writer.write(self.row_number, 1, f'{exp_loss_face.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 2, f'{exp_loss_edge.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 3, f'{exp_loss_chamfer.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 4, f'{exp_completeness.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 5, f'{map_loss_face.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 6, f'{map_loss_edge.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 7, f'{map_loss_chamfer.detach().cpu().numpy():.3f}')
            self.row_number += 1
            self.wb.save(self.xls_file)

        print('\n')
        rospy.loginfo(f'Exploration Edge loss: {exp_loss_edge.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration Face loss: {exp_loss_face.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration Chamfer loss: {exp_loss_chamfer.squeeze().detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration completeness: {exp_completeness.detach().cpu().numpy()}')
        rospy.loginfo(f'Num covered points: {int(coverage_mask.sum())}/{map_gt.size()[1]}')
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
        self.exp_compl_pub.publish(Float64(exp_completeness.detach().cpu().numpy()))

        self.map_face_acc_pub.publish(Float64(map_loss_face.detach().cpu().numpy()))
        self.map_edge_acc_pub.publish(Float64(map_loss_edge.detach().cpu().numpy()))
        self.map_chamfer_acc_pub.publish(Float64(map_loss_chamfer.squeeze().detach().cpu().numpy()))
        return coverage_mask

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            self.map_gt_mesh_marker.header.stamp = rospy.Time.now()
            self.world_mesh_pub.publish(self.map_gt_mesh_marker)
            if self.artifacts_cloud is not None:
                self.publish_pointcloud(self.artifacts_cloud, 'artifacts_cloud', rospy.Time(0), self.map_gt_frame)
            if self.pc_msg is None:
                continue

            if self.map_gt is not None and self.map is not None:
                # compare subscribed map point cloud to ground truth mesh
                if self.do_eval:
                    self.coverage_mask = self.eval(map=self.map,
                                                   map_gt=self.map_gt,
                                                   map_gt_mesh=self.map_gt_mesh,
                                                   coverage_dist_th=self.coverage_dist_th)

                # publish point cloud from ground truth mesh
                points = self.map_gt  # (1, N, 3)
                if self.coverage_mask is not None:
                    assert self.coverage_mask.shape[:2] == self.map_gt.shape[:2]
                    points = torch.cat([points, self.coverage_mask], dim=2)

                    # selecting default map frame if it is not available in ROS
                    map_gt_frame = self.map_gt_frame if self.map_gt_frame is not None else 'subt'
                    points_to_pub = points.squeeze().cpu().numpy().transpose(1, 0)
                    assert len(points_to_pub.shape) == 2
                    assert points_to_pub.shape[0] >= 3
                    self.publish_pointcloud(points_to_pub,
                                            topic_name='~cloud_from_gt_mesh',
                                            stamp=rospy.Time.now(),
                                            frame_id=map_gt_frame)
                    rospy.logdebug(f'Ground truth mesh frame: {map_gt_frame}')
                    rospy.logdebug(f'Publishing points of shape {points_to_pub.shape} sampled from ground truth mesh')
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('map_eval', log_level=rospy.INFO)
    path_fo_gt_map_mesh = rospy.get_param('~gt_mesh')
    assert os.path.exists(path_fo_gt_map_mesh)
    rospy.loginfo('Using ground truth mesh file: %s', path_fo_gt_map_mesh)
    proc = MapEval(map_topic=rospy.get_param('~map_topic', 'map_accumulator/map'),
                   path_to_obj=path_fo_gt_map_mesh)
    rospy.loginfo('Mapping evaluation node is initialized. You may need to heat Space for bagfile to start.')
    proc.run()
