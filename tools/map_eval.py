#!/usr/bin/env python

import os

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
from sensor_msgs.msg import PointCloud2, PointCloud
from std_msgs.msg import Float64
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
import trimesh
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
    def __init__(self, map_topic, path_to_mesh):
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)
        # Set the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.seed = 0  # for reprodusibility of experiments
        # parameters
        self.do_points_sampling_from_mesh = rospy.get_param('~do_points_sampling_from_mesh', True)
        self.do_points_sampling_from_map = rospy.get_param('~do_points_sampling_from_map', True)
        self.n_sample_points = rospy.get_param('~n_sample_points', 10000)
        self.do_eval = rospy.get_param('~do_eval', True)
        self.load_gt = rospy.get_param('~load_gt', True)
        self.record_metrics = rospy.get_param('~record_metrics', True)
        self.xls_file = rospy.get_param('~metrics_xls_file', f'/tmp/mapping-eval')
        self.xls_file = f'{self.xls_file[:-4]}.xls'
        self.coverage_dist_th = rospy.get_param('~coverage_dist_th', 0.2)
        self.artifacts_coverage_dist_th = rospy.get_param('~artifacts_coverage_dist_th', 0.1)
        self.artifacts_hypothesis_topic = rospy.get_param('~artifacts_hypothesis_topic',
                                                          'detection_localization/dbg_confirmed_hypotheses_pcl')
        self.max_age = rospy.get_param('~max_age', 1.0)
        self.rate = rospy.get_param('~eval_rate', 1.0)
        # exploration metrics publishers
        self.face_loss_pub = rospy.Publisher('~exp_loss_face', Float64, queue_size=1)
        self.edge_loss_pub = rospy.Publisher('~exp_loss_edge', Float64, queue_size=1)
        self.chamfer_loss_pub = rospy.Publisher('~exp_loss_chamfer', Float64, queue_size=1)
        self.exp_compl_pub = rospy.Publisher('~exp_completeness', Float64, queue_size=1)
        self.artif_exp_compl_pub = rospy.Publisher('~artifacts_exp_completeness', Float64, queue_size=1)
        # mapping accuracy publishers
        self.map_face_acc_pub = rospy.Publisher('~map_loss_face', Float64, queue_size=1)
        self.map_edge_acc_pub = rospy.Publisher('~map_loss_edge', Float64, queue_size=1)
        self.map_chamfer_acc_pub = rospy.Publisher('~map_loss_chamfer', Float64, queue_size=1)
        # artifacts detections score publisher
        self.dets_score_pub = rospy.Publisher('~detections_score', Float64, queue_size=1)
        self.artifacts_names = ["rescue_randy", "phone", "backpack", "drill", "extinguisher",
                                "vent", "helmet", "rope", "cube"]
        # world ground truth mesh publisher
        self.world_mesh_pub = rospy.Publisher('/world_mesh', Marker, queue_size=1)
        # total reward publisher
        self.reward_pub = rospy.Publisher('~reward', Float64, queue_size=1)

        self.map_gt_frame = rospy.get_param('~map_gt_frame', 'subt')
        self.map_gt_mesh = None
        self.map_gt_trimesh = None  # mesh loaded with trimesh library
        self.map_gt_mesh_marker = Marker()
        self.artifacts = {'poses': None, 'classes': None, 'clouds': None, 'cloud_merged': None}
        self.map_gt = None
        self.pc_msg = None
        self.map_frame = None
        self.map = None
        self.detections = {'poses': None, 'classes': None}

        # loading ground truth data
        if self.load_gt:
            self.load_ground_truth(path_to_mesh)

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
            self.ws_writer.write(0, 8, 'Artifacts Exploration completeness')
            self.ws_writer.write(0, 9, 'Detections score')
            self.ws_writer.write(0, 10, 'N of constructed points')
            self.ws_writer.write(0, 11, 'Total reward')
            self.row_number = 1

        # obtaining the constructed map (reward cloud)
        self.map_sub = rospy.Subscriber(map_topic, PointCloud2, self.get_constructed_map)
        # subscribing to localized detection results for evaluation
        rospy.Subscriber(self.artifacts_hypothesis_topic, PointCloud, self.get_detections)

    @staticmethod
    def publish_pointcloud(points, topic_name, stamp, frame_id, intensity='i'):
        if points is None:
            rospy.logwarn('Point cloud is None, not published')
            return
        assert isinstance(points, np.ndarray)
        assert len(points.shape) == 2
        assert points.shape[0] >= 3
        # create PointCloud2 msg
        data = np.zeros(points.shape[1], dtype=[('x', np.float32),
                                                ('y', np.float32),
                                                ('z', np.float32),
                                                (intensity, np.float32)])
        data['x'] = points[0, ...]
        data['y'] = points[1, ...]
        data['z'] = points[2, ...]
        if points.shape[0] > 3:
            data[intensity] = points[3, ...]
        pc_msg = msgify(PointCloud2, data)
        pc_msg.header.stamp = stamp
        pc_msg.header.frame_id = frame_id
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def load_ground_truth(self, path_to_mesh_file):
        assert os.path.exists(path_to_mesh_file)
        t0 = timer()
        rospy.loginfo('Loading ground truth mesh ...')
        if '.obj' in path_to_mesh_file:
            gt_mesh_verts, faces, _ = load_obj(path_to_mesh_file)
            gt_mesh_faces_idx = faces.verts_idx
            # self.map_gt_trimesh = trimesh.load(path_to_mesh_file)
        elif '.ply' in path_to_mesh_file:
            gt_mesh_verts, gt_mesh_faces_idx = load_ply(path_to_mesh_file)
            # self.map_gt_trimesh = trimesh.load(path_to_mesh_file)
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
        if self.do_points_sampling_from_mesh:
            torch.manual_seed(self.seed)
            self.map_gt = sample_points_from_meshes(self.map_gt_mesh, self.n_sample_points)
        else:
            self.map_gt = gt_mesh_verts.unsqueeze(0)
        self.map_gt = self.map_gt.to(self.device)
        assert self.map_gt.dim() == 3
        rospy.loginfo(f'Loaded mesh with verts shape: {gt_mesh_verts.size()} in {(timer() - t0):.3f} [sec]')

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
        self.artifacts = self.get_artifacts()

    def get_artifacts(self, frames_lookup_time=10.0):
        # get artifacts list from tf tree
        all_frames = []
        artifact_frames = []
        artifact_names = ['backpack', 'black_and_decker_cordless_drill',
                          'climbing_helmet_with_light.mtl', 'climbing_rope', 'fire_extinguisher',
                          'gas', 'phone', 'rescue_randy']
        t0 = timer()
        rospy.loginfo('Looking for artifacts ...')
        while len(artifact_frames) == 0 and (timer() - t0) < frames_lookup_time:
            all_frames = self.tf._getFrameStrings()
            for frame in all_frames:
                for name in artifact_names:
                    if name in frame:
                        artifact_frames.append(frame)
        rospy.logdebug('Found TF frames: %s', all_frames)
        rospy.loginfo('Found Artifacts TF frames in %.3f [sec]: %s', (timer() - t0), artifact_frames)
        # artifact_frames = ['backpack_1', 'backpack_2', 'backpack_3', 'backpack_4',
        #              'phone_1', 'phone_2', 'phone_3', 'phone_4',
        #              'rescue_randy_1', 'rescue_randy_2', 'rescue_randy_3', 'rescue_randy_4']
        artifacts = {'poses': [], 'classes': [], 'clouds': [], 'cloud_merged': None}
        artifacts_cloud_merged = []
        for i, artifact_name in enumerate(artifact_frames):
            try:
                artifact_frame = artifact_name
                transform = self.tf.lookup_transform(self.map_gt_frame, artifact_frame, rospy.Time(0))
            except tf2_ros.LookupException:
                rospy.logwarn('Ground truth artifacts poses are not available')
                return artifacts

            # create artifacts point cloud here from their meshes
            verts, faces, _ = load_obj(os.path.join(rospkg.RosPack().get_path('rpz_planning'),
                                                    f"data/meshes/artifacts/{artifact_name[:-2]}.obj"))
            # mesh = Meshes(verts=[verts], faces=[faces.verts_idx]).to(self.device)
            # torch.manual_seed(self.seed); cloud = sample_points_from_meshes(mesh, 1000).squeeze(0).to(self.device)
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
            # add intensity value based on the artifact type
            if 'backpack' in artifact_name:
                intensity = 1
            elif 'phone' in artifact_name:
                intensity = 2
            elif 'rescue_randy' in artifact_name:
                intensity = 3
            else:
                intensity = -1
            cloud = np.concatenate([cloud, intensity * np.ones([1, cloud.shape[1]])], axis=0)
            artifacts_cloud_merged.append(cloud)
            artifacts['classes'].append(artifact_name[:-2])  # without _n at the end of the name
            artifacts['poses'].append(t)
            artifacts['clouds'].append(torch.as_tensor(cloud.transpose([1, 0]), dtype=torch.float32).unsqueeze(0).to(self.device))
        artifacts_cloud_merged_np = artifacts_cloud_merged[0]
        for cloud in artifacts_cloud_merged[1:]:
            artifacts_cloud_merged_np = np.concatenate([artifacts_cloud_merged_np, cloud], axis=1)
        assert artifacts_cloud_merged_np.shape[0] == 4
        artifacts['cloud_merged'] = artifacts_cloud_merged_np
        return artifacts

    def get_constructed_map(self, pc_msg):
        assert isinstance(pc_msg, PointCloud2)
        rospy.logdebug('Received point cloud message')
        t0 = timer()
        self.pc_msg = pc_msg
        self.map_frame = pc_msg.header.frame_id
        if self.map_gt_frame is None:
            rospy.logwarn('Ground truth map frame is not yet available')
            return
        try:
            transform = self.tf.lookup_transform(self.map_gt_frame, self.map_frame, rospy.Time(0))
        except tf2_ros.LookupException:
            rospy.logwarn('No transform between constructed map frame and its ground truth')
            return
        map = numpify(self.pc_msg)
        map = np.vstack([map[f] for f in map.dtype.fields])  # ['x', 'y', 'z', ('reward')]
        T = numpify(transform.transform)
        R, t = T[:3, :3], T[:3, 3]
        assert map.shape[0] >= 3  # (>=3, N)
        map[:3, ...] = np.matmul(R, map[:3, ...]) + t.reshape([3, 1])

        self.map = torch.as_tensor(map.transpose([1, 0]), dtype=torch.float32).unsqueeze(0).to(self.device)
        assert self.map.dim() == 3
        assert self.map.size()[2] >= 3
        rospy.logdebug('Point cloud conversion took: %.3f s', timer() - t0)

    def get_detections(self, pc_msg):
        assert isinstance(pc_msg, PointCloud)
        poses = []
        for p in pc_msg.points:
            poses.append(np.array([p.x, p.y, p.z]))
        self.detections['poses'] = np.array(poses)
        class_numbers = pc_msg.channels[-1].values  # most probable class values
        # convert numbers to names
        self.detections['classes'] = [self.artifacts_names[int(n)] for n in class_numbers]

    def evaluate_detections(self, preds, gts, dist_th=1.0):
        # TODO: check implementation of the detections accuracy here
        # the final score should also include false positives and true negatives
        # compute precision and recall:
        # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
        assert isinstance(preds, dict)
        assert isinstance(gts, dict)
        poses = torch.as_tensor(preds['poses']).to(self.device)
        poses_gt = torch.as_tensor(gts['poses']).to(self.device)
        assert poses.shape[1] == poses_gt.shape[1]
        knn = knn_points(poses[None], poses_gt[None], K=1)
        # currently the score just takes into account proximity of the predictions to ground truth
        # by a distance threshold and, if the predicted class matches ground truth the score is increased.
        # It is weighted by the number of predictions to mitigate spamming a lot of false/random predictions
        score = 0.0
        for i, d in enumerate(knn.dists.squeeze(0)):
            if d <= dist_th:
                if gts['classes'][i] == preds['classes'][knn.idx.squeeze()[i]]:
                    score += 1
        score /= len(preds['classes'])
        return score

    @staticmethod
    def estimate_coverage(cloud, map_gt_cloud, coverage_dist_th=0.2):
        if cloud is None:
            rospy.logwarn('Rewards cloud is not yet received')
            return None
        if map_gt_cloud is None:
            rospy.logwarn('Ground truth map cloud is not yet received')
            return None
        assert isinstance(cloud, torch.Tensor)
        assert cloud.dim() == 3
        assert cloud.size()[2] >= 3  # (1, N1, 3(4))
        assert isinstance(map_gt_cloud, torch.Tensor)
        assert map_gt_cloud.dim() == 3
        assert map_gt_cloud.size()[2] == 3  # (1, N2, 3)
        map = cloud[..., :3]
        map_nn = knn_points(p1=map_gt_cloud, p2=map, K=1)
        coverage_mask = map_nn.dists.sqrt() < coverage_dist_th
        assert coverage_mask.shape[:2] == map_gt_cloud.shape[:2]
        exp_completeness = coverage_mask.sum() / map_gt_cloud.size()[1]

        if cloud.shape[2] > 3:
            # include rewards information in coverage mask computation
            rewards = cloud[..., 3].unsqueeze(2)
            rewards_mask = coverage_mask * rewards[:, map_nn.idx.squeeze(), :]
            assert rewards_mask.shape[:2] == map_gt_cloud.shape[:2]
            return exp_completeness, rewards_mask
        else:
            return exp_completeness, coverage_mask

    def evaluation(self, map, map_gt_cloud, map_gt_mesh, artifacts,
                   coverage_dist_th=0.2,
                   artifacts_coverage_dist_th=0.1):
        if map is None:
            rospy.logwarn('Evaluation: Map cloud is not yet received')
            return None
        if map_gt_cloud is None:
            rospy.logwarn('Evaluation: Ground truth map cloud is not yet received')
            return None
        if map_gt_mesh is None:
            rospy.logwarn('Evaluation: Ground truth map mesh is not yet received')
            return None
        if artifacts is None:
            rospy.logwarn('Evaluation: Artifacts cloud is not yet received')
            return None
        assert isinstance(map, torch.Tensor)
        assert map.dim() == 3
        assert map.size()[2] >= 3  # (1, N1, >=3)
        assert isinstance(map_gt_cloud, torch.Tensor)
        assert map_gt_cloud.dim() == 3
        assert map_gt_cloud.size()[2] == 3  # (1, N2, 3)
        assert isinstance(artifacts, dict)
        assert isinstance(artifacts['clouds'], list)
        assert isinstance(artifacts['classes'], list)
        assert isinstance(artifacts['cloud_merged'], np.ndarray)
        for cloud in artifacts['clouds']:
            assert isinstance(cloud, torch.Tensor)
            assert cloud.dim() == 3
            assert cloud.size()[2] >= 3  # (1, n, >=3)
        # Discard old messages.
        time_stamp = rospy.Time.now()
        age = (time_stamp - self.pc_msg.header.stamp).to_sec()
        if age > self.max_age:
            rospy.logwarn('Evaluation: Discarding points %.1f s > %.1f s old.', age, self.max_age)
            return None
        rospy.loginfo(f'Received map of size {map.size()} for evaluation...')
        # compare point cloud to mesh here
        with torch.no_grad():
            t1 = timer()
            map_sampled = map.clone()
            if self.do_points_sampling_from_map:
                if self.n_sample_points < map.shape[1]:
                    torch.manual_seed(self.seed)
                    map_sampled = map[:, torch.randint(map.shape[1], (self.n_sample_points,)), :]
            # self.publish_pointcloud(map_sampled.squeeze(0).transpose(1, 0).detach().cpu().numpy(),
            #                         '~map_sampled', rospy.Time.now(), self.map_gt_frame)
            N_map_points = map_sampled.shape[1]
            point_cloud = Pointclouds(map_sampled[..., :3]).to(self.device)

            exp_loss_edge = edge_point_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            # distance between mesh and points is computed as a distance from point to triangle
            # if point's projection is inside triangle, then the distance is computed along
            # a normal to triangular plane. Otherwise as a distance to closest edge of the triangle:
            # https://github.com/facebookresearch/pytorch3d/blob/fe39cc7b806afeabe64593e154bfee7b4153f76f/pytorch3d/csrc/utils/geometry_utils.h#L635
            exp_loss_face = face_point_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            # exp_loss_face_trimesh = self.map_gt_trimesh.nearest.on_surface(map_sampled.squeeze().detach().cpu())[1].mean()
            # rospy.loginfo(f'Trimesh Exploration Face loss: {exp_loss_face_trimesh:.3f}')
            exp_loss_chamfer = chamfer_distance_truncated(x=map_gt_cloud, y=map[..., :3])

            t2 = timer()
            rospy.logdebug('Explored space evaluation took: %.3f s\n', t2 - t1)

            # coverage metric (exploration completeness)
            # The metric is evaluated as the fraction of ground truth points considered as covered.
            # A ground truth point is considered as covered if its nearest neighbour
            # there is a point from the constructed map that is located within a distance threshold
            # from the ground truth point.
            exp_completeness, coverage_mask = self.estimate_coverage(map, map_gt_cloud,
                                                                     coverage_dist_th=coverage_dist_th)
            # total reward for the whole exploration route based on visibility information:
            #   - 1-5m range
            #   - occlusion
            #   - cameras fov
            total_reward = -1
            if map.shape[2] == 4:  # if map contains reward values
                total_reward = coverage_mask.sum().detach()
                rospy.loginfo(f'Total reward: {total_reward:.3f}')
                self.reward_pub.publish(Float64(total_reward))

            t3 = timer()
            rospy.logdebug('Coverage estimation took: %.3f s\n', t3 - t2)

            # compute the same metric for each individual artifact
            artifacts_exp_completeness = 0.0
            for i, cloud in enumerate(artifacts['clouds']):
                # number of artifact points that are covered
                cloud = cloud[..., :3]
                assert cloud.shape[2] == map[..., :3].shape[2]
                map_nn = knn_points(p1=cloud, p2=map[..., :3], K=1)
                artifact_coverage_mask = map_nn.dists.sqrt() < artifacts_coverage_dist_th
                assert artifact_coverage_mask.shape[:2] == cloud.shape[:2]
                if artifact_coverage_mask.sum() > 0:
                    artifacts_exp_completeness += artifact_coverage_mask.sum()
                    rospy.loginfo(f'Explored {int(artifact_coverage_mask.sum())} / {cloud.shape[1]} '
                                  f'points of artifact {artifacts["classes"][i]}')
            artifacts_exp_completeness = artifacts_exp_completeness / (artifacts['cloud_merged'].shape[1])

            dets_score = 0.0
            if self.detections['poses'] is not None:
                dets_score = self.evaluate_detections(self.detections, self.artifacts, dist_th=1.0)

            t4 = timer()
            rospy.logdebug('Artifacts evaluation took: %.3f s\n', t4 - t3)

            # `exp_loss_face`, `exp_loss_edge` and `exp_loss_chamfer` describe exploration progress
            # current map accuracy could be evaluated by computing vice versa distances:
            # - from points in cloud to mesh faces/edges:
            map_loss_edge = point_edge_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            map_loss_face = point_face_distance_truncated(meshes=map_gt_mesh, pcls=point_cloud)
            # - from points in cloud to nearest neighbours of points sampled from mesh:
            map_loss_chamfer = chamfer_distance_truncated(x=map[..., :3], y=map_gt_cloud,
                                                          apply_point_reduction=True,
                                                          batch_reduction='mean', point_reduction='mean')
            t5 = timer()
            rospy.logdebug('Mapping accuracy evaluation took: %.3f s\n', t5 - t4)
            rospy.loginfo('Evaluation took: %.3f s\n', t5 - t1)

        # record data
        if self.record_metrics:
            self.ws_writer.write(self.row_number, 0, time_stamp.to_sec())
            self.ws_writer.write(self.row_number, 1, f'{exp_loss_face.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 2, f'{exp_loss_edge.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 3, f'{exp_loss_chamfer.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 4, f'{exp_completeness.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 5, f'{map_loss_face.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 6, f'{map_loss_edge.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 7, f'{map_loss_chamfer.detach().cpu().numpy():.3f}')
            self.ws_writer.write(self.row_number, 8, f'{artifacts_exp_completeness:.3f}')
            self.ws_writer.write(self.row_number, 9, f'{dets_score:.3f}')
            self.ws_writer.write(self.row_number, 10, f'{N_map_points}')
            self.ws_writer.write(self.row_number, 11, f'{total_reward}')
            self.row_number += 1
            self.wb.save(self.xls_file)

        print('\n')
        rospy.loginfo(f'Number of points in the constructed map: {N_map_points}')
        rospy.loginfo(f'Exploration Edge loss: {exp_loss_edge.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration Face loss: {exp_loss_face.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration Chamfer loss: {exp_loss_chamfer.squeeze().detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Exploration completeness: {exp_completeness.detach().cpu().numpy():.3f}')
        rospy.loginfo(f'Artifacts exploration completeness: {artifacts_exp_completeness:.3f}')
        rospy.loginfo(f'Detections score: {dets_score}')
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
        self.artif_exp_compl_pub.publish(Float64(artifacts_exp_completeness))
        self.dets_score_pub.publish(Float64(dets_score))

        self.map_face_acc_pub.publish(Float64(map_loss_face.detach().cpu().numpy()))
        self.map_edge_acc_pub.publish(Float64(map_loss_edge.detach().cpu().numpy()))
        self.map_chamfer_acc_pub.publish(Float64(map_loss_chamfer.squeeze().detach().cpu().numpy()))
        return coverage_mask

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            self.map_gt_mesh_marker.header.stamp = rospy.Time.now()
            self.world_mesh_pub.publish(self.map_gt_mesh_marker)
            self.publish_pointcloud(self.artifacts['cloud_merged'], 'artifacts_cloud', rospy.Time.now(), self.map_gt_frame)

            # compare subscribed map point cloud to ground truth mesh
            # self.publish_pointcloud(self.map.squeeze(0).transpose(1, 0).detach().cpu().numpy(),
            #                         'map', rospy.Time.now(), self.map_gt_frame)
            if self.do_eval:
                coverage_mask = self.evaluation(map=self.map,
                                                map_gt_cloud=self.map_gt,
                                                map_gt_mesh=self.map_gt_mesh,
                                                artifacts=self.artifacts,
                                                coverage_dist_th=self.coverage_dist_th,
                                                artifacts_coverage_dist_th=self.artifacts_coverage_dist_th)

                # publish point cloud from ground truth mesh
                if coverage_mask is not None:
                    binary_coverage_mask = coverage_mask > 0
                    assert binary_coverage_mask.shape[:2] == self.map_gt.shape[:2]
                    points = self.map_gt  # (1, N, 3)
                    points = torch.cat([points, binary_coverage_mask], dim=2)

                    # selecting default map frame if it is not available in ROS
                    map_gt_frame = self.map_gt_frame if self.map_gt_frame is not None else 'subt'
                    points_to_pub = points.squeeze().cpu().numpy().transpose(1, 0)
                    assert len(points_to_pub.shape) == 2
                    assert points_to_pub.shape[0] >= 3
                    self.publish_pointcloud(points_to_pub,
                                            topic_name='~cloud_from_gt_mesh',
                                            stamp=rospy.Time.now(),
                                            frame_id=map_gt_frame,
                                            intensity='coverage')
                    rospy.logdebug(f'Ground truth mesh frame: {map_gt_frame}')
                    rospy.logdebug(f'Publishing points of shape {points_to_pub.shape} sampled from ground truth mesh')
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('map_eval', log_level=rospy.DEBUG)
    path_fo_gt_map_mesh = rospy.get_param('~gt_mesh')
    assert os.path.exists(path_fo_gt_map_mesh)
    rospy.loginfo('Using ground truth mesh file: %s', path_fo_gt_map_mesh)
    proc = MapEval(map_topic=rospy.get_param('~map_topic', 'map_accumulator/map'),
                   path_to_mesh=path_fo_gt_map_mesh)
    rospy.loginfo('Mapping evaluation node is initialized.')
    rospy.loginfo('You may need to heat Space for bagfile to start.')
    proc.run()
