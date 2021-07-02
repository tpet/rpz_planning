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


class GTWorldPub:
    """
    This ROS node publishes ground truth world (mesh and point clouds).
    """
    def __init__(self, world_name='simple_cave_01'):
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
        self.n_sample_points = rospy.get_param('~n_sample_points', 10000)
        # world ground truth mesh publisher
        self.world_mesh_pub = rospy.Publisher('/world_mesh', Marker, queue_size=1)

        self.map_gt_frame = world_name
        self.map_gt_mesh = None
        self.map_gt_mesh_marker = Marker()
        self.artifacts = {'poses': None, 'classes': None, 'clouds': None, 'cloud_merged': None}
        self.map_gt = None
        self.rate = rospy.get_param('~rate', 1.0)

        # loading ground truth data
        self.load_ground_truth(world_name)

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

    def load_ground_truth(self, world_name):
        path_to_mesh_file = os.path.join(rospkg.RosPack().get_path('rpz_planning'), f"data/meshes/{world_name}.obj")
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

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            # publish ground truth
            self.map_gt_mesh_marker.header.stamp = rospy.Time.now()
            self.world_mesh_pub.publish(self.map_gt_mesh_marker)
            self.publish_pointcloud(self.artifacts['cloud_merged'], 'artifacts_cloud', rospy.Time.now(), self.map_gt_frame)

            # selecting default map frame if it is not available in ROS
            points_to_pub = self.map_gt.squeeze().cpu().numpy().transpose(1, 0)
            assert len(points_to_pub.shape) == 2
            assert points_to_pub.shape[0] >= 3
            self.publish_pointcloud(points_to_pub,
                                    topic_name='/cloud_from_gt_mesh',
                                    stamp=rospy.Time.now(),
                                    frame_id=self.map_gt_frame,
                                    intensity='coverage')
            rospy.logdebug(f'Ground truth mesh frame: {self.map_gt_frame}')
            rospy.logdebug(f'Publishing points of shape {points_to_pub.shape} sampled from ground truth mesh')
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('ground_truth_world_publisher', log_level=rospy.INFO)
    world_name = rospy.get_param('/world_name', 'simple_cave_01')
    rospy.loginfo('Loading world: %s', world_name)
    proc = GTWorldPub(world_name=world_name)
    rospy.loginfo('Ground truth publisher node is initialized.')
    proc.run()
