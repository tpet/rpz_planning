#!/usr/bin/env python

import os
import torch
from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
import rospy
from sensor_msgs.msg import PointCloud2, PointCloud
import numpy as np
from ros_numpy import msgify, numpify
import tf2_ros
from timeit import default_timer as timer
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
from scipy.spatial.transform import Rotation
import seaborn as sns
# problems with Cryptodome: pip install pycryptodomex
# https://github.com/DP-3T/reference_implementation/issues/1


class GTWorldPub:
    """
    This ROS node publishes ground truth world (mesh and point clouds).
    """

    def __init__(self):
        self.tf = tf2_ros.Buffer()
        self.tl = tf2_ros.TransformListener(self.tf)
        # Set the device
        self.device = torch.device("cpu")
        self.seed = 0  # for reprodusibility of experiments
        # parameters
        self.world_name = rospy.get_param('/world_name', 'simple_cave_03')
        rospy.loginfo('Loading world: %s', self.world_name)
        self.do_points_sampling_from_mesh = rospy.get_param('~do_points_sampling_from_mesh', True)
        self.n_sample_points = rospy.get_param('~n_sample_points', 10000)
        # world ground truth mesh publisher
        self.world_mesh_pub = rospy.Publisher('/world_mesh', Marker, queue_size=1)
        self.artifacts_meshes_pub = rospy.Publisher('/artifacts_meshes', MarkerArray, queue_size=1)
        self.map_gt_frame = self.world_name
        self.robot_gt_frame = 'X1_ground_truth'
        self.map_gt_mesh = None
        self.map_gt_mesh_marker = Marker()
        self.artifacts_gt_marker_array = MarkerArray()
        self.artifacts = {'poses': None, 'classes': None, 'clouds': None, 'cloud_merged': None}
        self.artifacts_names = ["rescue_randy", "phone", "backpack", "drill", "extinguisher",
                                "vent", "helmet", "rope", "cube"]
        self.rgb_colors_palette = sns.color_palette(None, len(self.artifacts_names))  # for visualization in RViz
        self.map_gt = None
        self.rate = rospy.get_param('~rate', 1.0)

        # currently supported ground truth meshes of worlds
        self.publish_world_mesh = self.world_name in ['simple_cave_01', 'simple_cave_02', 'simple_cave_03']
        if self.publish_world_mesh:
            self.load_world_mesh()
        # get the artifacts point cloud
        self.artifacts = self.get_artifacts()

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
        data['z'] = points[2, ...] + 0.5
        if points.shape[0] > 3:
            data[intensity] = points[3, ...]
        pc_msg = msgify(PointCloud2, data)
        pc_msg.header.stamp = stamp
        pc_msg.header.frame_id = frame_id
        pub = rospy.Publisher(topic_name, PointCloud2, queue_size=1)
        pub.publish(pc_msg)

    def load_world_mesh(self):
        path_to_mesh_file = os.path.join(rospkg.RosPack().get_path('rpz_planning'), f"data/meshes/{self.world_name}.obj")
        assert os.path.exists(path_to_mesh_file)
        t0 = timer()
        rospy.loginfo('Loading ground truth mesh ...')
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
        R = torch.tensor([[0., 1., 0.],
                          [-1., 0., 0.],
                          [0., 0., 1.]]).to(self.device)
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
        marker.ns = "world_ns"
        marker.id = 0
        marker.action = Marker.ADD
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        r = np.array([[0., 0., 1.],
                      [-1., 0., 0.],
                      [0., -1., 0.]])
        q = Rotation.from_matrix(r).as_quat()
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 0.4
        marker.color.r = 0.4
        marker.color.g = 0.5
        marker.color.b = 0.6
        marker.type = Marker.MESH_RESOURCE
        marker.mesh_resource = f"package://rpz_planning/data/meshes/{self.world_name}.dae"
        self.map_gt_mesh_marker = marker

    def get_artifacts(self, frames_lookup_time=3.0):
        # get artifacts list from tf tree
        all_frames = []
        artifact_frames = []

        # time.sleep(frames_lookup_time)

        t0 = timer()
        rospy.loginfo('Looking for artifacts ...')
        while len(artifact_frames) == 0 and (timer() - t0) < frames_lookup_time:
            all_frames = self.tf._getFrameStrings()
            for frame in all_frames:
                for name in self.artifacts_names:
                    if name in frame:
                        artifact_frames.append(frame)
        rospy.logdebug('Found TF frames: %s', all_frames)
        rospy.loginfo('Found %i Artifacts TF frames in %.3f [sec]: %s',
                      len(artifact_frames), (timer() - t0), artifact_frames)

        artifacts = {'poses': [], 'names': [], 'clouds': [], 'cloud_merged': None}
        artifacts_cloud_merged = []
        for i, artifact_name in enumerate(artifact_frames):
            if 'gas' in artifact_name:
                break
            try:
                artifact_frame = artifact_name
                transform = self.tf.lookup_transform(self.map_gt_frame, artifact_frame, rospy.Time(0),
                                                     rospy.Duration(3))
            except tf2_ros.LookupException:
                rospy.logerr('Ground truth artifacts poses are not available')
                return artifacts
            T = numpify(transform.transform)
            R, t = T[:3, :3], T[:3, 3]
            # TODO: correct coordinates mismatch in Blender (swap here X and Y)
            R0 = np.array([[0., -1., 0.],
                           [1., 0., 0.],
                           [0., 0., 1.]])

            # construct visualization marker for each gt artifact
            marker = Marker()
            marker.header.frame_id = self.map_gt_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "artifacts_ns"
            marker.id = i
            marker.action = Marker.ADD
            marker.pose.position.x = t[0]
            marker.pose.position.y = t[1]
            marker.pose.position.z = t[2]
            q = Rotation.from_matrix(R @ R0).as_quat()
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.color.a = 0.9
            color_ind = self.artifacts_names.index(artifact_name[:-2])
            marker.color.r = self.rgb_colors_palette[color_ind][0]
            marker.color.g = self.rgb_colors_palette[color_ind][1]
            marker.color.b = self.rgb_colors_palette[color_ind][2]
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = f"package://rpz_planning/data/meshes/artifacts/{artifact_name[:-2]}.dae"
            self.artifacts_gt_marker_array.markers.append(marker)

            # create artifacts point cloud here from their meshes
            # verts, faces, _ = load_obj(os.path.join(rospkg.RosPack().get_path('rpz_planning'),
            #                                         f"data/meshes/artifacts/{artifact_name[:-2]}.obj"))
            verts, faces, _ = load_obj(os.path.join(rospkg.RosPack().get_path('rpz_planning'),
                                                    "data/meshes/artifacts/rescue_randy.obj"))
            # mesh = Meshes(verts=[verts], faces=[faces.verts_idx]).to(self.device)
            # torch.manual_seed(self.seed)
            # cloud = sample_points_from_meshes(mesh, 1000).squeeze(0).to(self.device)
            cloud = verts.cpu().numpy().transpose(1, 0)
            # transform point cloud to global world frame
            T = numpify(transform.transform)
            R, t = T[:3, :3], T[:3, 3]
            cloud = np.matmul(R, cloud) + t.reshape([3, 1])
            # add intensity value based on the artifact type
            intensity = self.artifacts_names.index(artifact_name[:-2])
            cloud = np.concatenate([cloud, intensity * np.ones([1, cloud.shape[1]])], axis=0)
            assert len(cloud.shape) == 2
            assert cloud.shape[0] == 4  # (4, n)
            artifacts_cloud_merged.append(cloud)
            artifacts['names'].append(artifact_name)
            # center of an artifact point cloud in map_gt_frame
            xyz = cloud[:3, :].mean(axis=1)
            artifacts['poses'].append(xyz)
            artifacts['clouds'].append(
                torch.as_tensor(cloud.transpose([1, 0]), dtype=torch.float32).unsqueeze(0).to(self.device))

        artifacts_cloud_merged_np = artifacts_cloud_merged[0]
        for cloud in artifacts_cloud_merged[1:]:
            artifacts_cloud_merged_np = np.concatenate([artifacts_cloud_merged_np, cloud], axis=1)
        assert artifacts_cloud_merged_np.shape[0] == 4
        artifacts['cloud_merged'] = artifacts_cloud_merged_np
        return artifacts

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            # publish ground truth mesh
            stamp = rospy.Time.now()
            if self.publish_world_mesh:
                self.map_gt_mesh_marker.header.stamp = stamp
                self.world_mesh_pub.publish(self.map_gt_mesh_marker)
            self.artifacts_meshes_pub.publish(self.artifacts_gt_marker_array)
            self.publish_pointcloud(self.artifacts['cloud_merged'], 'artifacts_cloud', rospy.Time.now(),
                                    self.map_gt_frame)

            # publish ground truth cloud
            gt_cloud = self.map_gt.squeeze().numpy().transpose(1, 0)
            assert len(gt_cloud.shape) == 2
            assert gt_cloud.shape[0] >= 3
            self.publish_pointcloud(gt_cloud,
                                    topic_name='/cloud_from_gt_mesh',
                                    stamp=stamp,
                                    frame_id=self.map_gt_frame,
                                    intensity='coverage')

            rospy.logdebug(f'Ground truth mesh frame: {self.map_gt_frame}')
            rospy.logdebug(f'Publishing points of shape {gt_cloud.shape} sampled from ground truth mesh')
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('ground_truth_world_publisher', log_level=rospy.INFO)
    proc = GTWorldPub()
    rospy.loginfo('Ground truth publisher node is initialized.')
    proc.run()
