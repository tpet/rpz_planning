#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import os.path
import rospkg
from geometry_msgs.msg import Point, Pose, PoseStamped, Pose2D, Quaternion, Transform, TransformStamped, Twist
import math
from matplotlib import colors
from nav_msgs.msg import Path
from std_msgs.msg import Float64
import numpy as np
import rospy
from ros_numpy import msgify, numpify
from rpz_planning import point_visibility
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler, euler_matrix
import tf2_ros
from threading import RLock
from timeit import default_timer as timer
import torch
import torch.nn.functional as fun
import xlwt
from visualization_msgs.msg import Marker, MarkerArray
from collections import OrderedDict
np.set_printoptions(precision=2)


def timing(f):
    def timing_wrapper(*args, **kwargs):
        t0 = timer()
        ret = f(*args, **kwargs)
        t1 = timer()
        rospy.logdebug('%s %.6f s' % (f.__name__, t1 - t0))
        return ret
    return timing_wrapper


class DimOrder(object):
    YAW_X_Y = 'YawXY'
    X_Y_YAW = 'XYYaw'


def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]


def array(msg):
    """Return message attributes (slots) as array."""
    return np.array(slots(msg))


def transform(tf, x):
    assert tf.shape[0] == tf.shape[1]
    assert tf.shape[0] == x.shape[0] + 1
    if isinstance(tf, np.ndarray):
        return np.matmul(tf[:-1, :-1], x) + tf[:-1, -1:]
    elif isinstance(tf, torch.Tensor):
        return torch.matmul(tf[:-1, :-1], x) + tf[:-1, -1:]
    raise TypeError('Invalid argument type: %s', type(tf))


@timing
def transform_cloud_msg(tf, msg):
    assert isinstance(tf, np.ndarray)
    assert isinstance(msg, PointCloud2)
    shape = msg.height, msg.width
    cloud = numpify(msg).ravel().copy()
    fields = 'x', 'y', 'z'
    x = np.stack([cloud[f] for f in fields])
    x = transform(tf, x)
    for i, f in enumerate(fields):
        cloud[f] = x[i]
    cloud = cloud.reshape(shape)
    out = msgify(PointCloud2, cloud)
    out.header = msg.header
    return out


def yaw_angles(msg):
    if isinstance(msg, PointCloud2):
        fields = [f.name for f in msg.fields]
    elif isinstance(msg, np.ndarray):
        fields = msg.dtype.names
        if fields is None:
            raise ValueError('Array dtype %s is not structured.' % msg.dtype)
    else:
        raise TypeError('Unsupported input type: %s.' % type(msg))

    angles = []
    for f in fields:
        if 'roll' in f:
            a = float(f.split('_')[1])
            angles.append(a)

    # FIXME: Allow any discretization.
    assert len(angles) == 8
    return angles


def map_affine(x, l0, h0, l1=0., h1=1.):
    """Affine map from interval [l0, h0] to [l1, h1].
    Broadcasting is used to match corresponding dimensions.
    """
    x = l1 + (x - l0) / (h0 - l0) * (h1 - l1)
    return x


@timing
def cloud_to_grid_transform(cloud, order=DimOrder.X_Y_YAW):
    """Estimate cloud-to-grid transform as 3x3 matrix.

    The matrix maps points (x, y) to normalized grid coordinates (u, v) from
    [-1, 1].

    The cloud dimensions should be (x, y), not (y, x).
    """
    assert isinstance(cloud, np.ndarray)
    assert cloud.ndim == 2

    assert order in (DimOrder.X_Y_YAW, DimOrder.YAW_X_Y)

    # Size of the grid
    w, h = cloud.shape

    A = []
    b = []

    for x, y in ((0, 0), (0, 1), (1, 0)):
        xy_cloud = [cloud['x'][x, y], cloud['y'][x, y]]
        xy_grid = [map_affine(x, -.5, w - .5, -1., 1.),
                   map_affine(y, -.5, h - .5, -1., 1.)]

        A.append([xy_cloud[0], xy_cloud[1], 1., 0., 0., 0.])
        A.append([0., 0., 0., xy_cloud[0], xy_cloud[1], 1.])
        b.append([xy_grid[0]])
        b.append([xy_grid[1]])

    A = torch.tensor(A)
    b = torch.tensor(b)
    sol, _ = torch.solve(b, A)

    cloud_to_grid = torch.cat((sol.reshape((2, 3)),
                               torch.tensor([[0., 0., 1.]])))
    # print('cloud_to_grid: %s' % (cloud_to_grid,))
    return cloud_to_grid


@timing
def cloud_msg_to_rpz_tensor(msg, order=DimOrder.X_Y_YAW):
    """Convert cloud message to RPZ tensor.

     Output size (RPZ=3, X, Y, YAW=8) or (RPZ=3, YAW, X, Y)."""
    assert isinstance(msg, PointCloud2)
    assert order in (DimOrder.X_Y_YAW, DimOrder.YAW_X_Y)
    rospy.logdebug('Converting RPZ cloud in %s of size (%i, %i) with channels %s...'
                   % (msg.header.frame_id, msg.height, msg.width, ', '.join([f.name for f in msg.fields])))

    # Positive x axis corresponds to columns,
    # which should be the shorter dimension.
    cld_sz = (250, 248)
    assert (msg.height * msg.width == cld_sz[0] * cld_sz[1])

    # TODO: Get yaw angles from channels?
    # yaws = list(range(0, 360, 45))
    yaws = yaw_angles(msg)
    assert len(yaws) == 8

    features = 'roll_%i', 'pitch_%i', 'z_%i'

    cloud = numpify(msg).reshape(cld_sz)
    # Convert to x, y dimensions instead of y, x.
    cloud = cloud.transpose(1, 0)
    cld_sz = cloud.shape

    # Convert cloud to RPZ tensor
    if order == DimOrder.YAW_X_Y:
        # of size (n_feat, n_yaw, 248, 250)
        rpz_sz = (len(features), len(yaws)) + cld_sz
        rpz = torch.full(rpz_sz, np.nan, dtype=torch.float32)
        for i, feat in enumerate(features):
            for j, yaw in enumerate(yaws):
                rpz[i, j, ...] = torch.tensor(cloud[feat % yaw], dtype=torch.float32)
    elif order == DimOrder.X_Y_YAW:
        # of size (n_feat, 248, 250, n_yaw)
        rpz_sz = (len(features),) + cld_sz + (len(yaws),)
        rpz = torch.full(rpz_sz, np.nan, dtype=torch.float32)
        for i, feat in enumerate(features):
            for j, yaw in enumerate(yaws):
                rpz[i, ..., j] = torch.tensor(cloud[feat % yaw], dtype=torch.float32)

    return rpz, cloud


@timing
def interpolate_rpz(rpz_all, xyyaw, order=DimOrder.YAW_X_Y, wrap_yaw=True):
    # TODO: Interpolate roll, pitch, z for given x, y, yaw.
    # TODO: https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
    # mode = 'bilinear' | 'nearest' | 'bicubic'
    # padding_mode = 'zeros' | 'border' | 'reflection'
    # Circular padding mode not supported.
    # TODO: Create YawD + 2 to wrap yaw for interpolation.
    assert isinstance(rpz_all, torch.Tensor)
    # (3, X, Y, Yaw) or (3, Yaw, X, Y)
    assert rpz_all.dim() == 4
    assert rpz_all.shape[0] == 3
    # print('rpz.shape: %s' % (rpz_all.shape,))

    assert isinstance(xyyaw, torch.Tensor)
    # (N, 3)
    assert xyyaw.dim() == 2
    assert xyyaw.shape[1] == 3

    assert order in (DimOrder.X_Y_YAW, DimOrder.YAW_X_Y)

    # Only bilinear in 5-D
    # mode = 'bilinear'

    if order == DimOrder.X_Y_YAW:
        assert rpz_all.shape[3] == 8
        if wrap_yaw:
            rpz_wrapped = torch.cat((rpz_all[..., -1:], rpz_all, rpz_all[..., :1]), dim=-1)
        else:
            rpz_wrapped = rpz_all
        xy_grid = xyyaw[:, :-1]
        yaw = xyyaw[:, -1:]
        n_yaw_steps = rpz_all.shape[-1]
    elif order == DimOrder.YAW_X_Y:
        assert rpz_all.shape[1] == 8
        if wrap_yaw:
            rpz_wrapped = torch.cat((rpz_all[:, -1:, ...], rpz_all, rpz_all[:, :1, ...]), dim=1)
        else:
            rpz_wrapped = rpz_all
        xy_grid = xyyaw[:, 1:]
        yaw = xyyaw[:, :1]
        n_yaw_steps = rpz_all.shape[1]
    yaw_step = 2. * np.pi / n_yaw_steps
    rospy.logdebug('rpz wrapped shape: %s', rpz_wrapped.shape)
    rospy.logdebug('yaw step: %.3f', yaw_step)

    # Without wrapping in [-1, 1].
    yaw_grid = torch.remainder(yaw + yaw_step / 2., 2. * np.pi) / np.pi - 1.
    # Squeeze due to extra wrapping bins.
    if wrap_yaw:
        yaw_grid = yaw_grid * (n_yaw_steps / (n_yaw_steps + 2))

    if order == DimOrder.YAW_X_Y:
        xyyaw_grid = torch.cat((yaw_grid, xy_grid), dim=1)
    elif order == DimOrder.X_Y_YAW:
        xyyaw_grid = torch.cat((xy_grid, yaw_grid), dim=1)

    # https://github.com/pytorch/pytorch/issues/35775#issuecomment-705702703
    # In addition to this, it seems like the order of the coordinates is also
    # inverted, i.e. for an (N, C, D, H, W) tensor, the (D, H, W) dimensions
    # correspond to (z, y, x) instead of (x, y, z).
    # xyyaw_grid = xyyaw_grid[..., ::-1]  # Unsupported
    xyyaw_grid = xyyaw_grid.flip(-1)

    rpz_wrapped = rpz_wrapped[None]
    xyyaw_grid = xyyaw_grid[None, None, None]
    assert torch.isnan(xyyaw_grid).any() == False
    rpz = fun.grid_sample(rpz_wrapped, xyyaw_grid,
                          padding_mode='border', align_corners=False)
    rpz = rpz[0, :, 0, 0, :].transpose(1, 0)
    return rpz


def pose_msg_to_xyzrpy(msg):
    assert isinstance(msg, Pose)
    xyz = slots(msg.position)
    rpy = euler_from_quaternion(slots(msg.orientation))
    return xyz + list(rpy)


def xyzrpy_to_pose_msg(xyzrpy):
    # assert isinstance(xyzrpy, torch.Tensor)
    # assert isinstance(xyzrpy, list)
    msg = Pose(Point(*xyzrpy[:3]), Quaternion(*quaternion_from_euler(*xyzrpy[3:])))
    return msg


@timing
def path_msg_to_xyzrpy(msg):
    assert isinstance(msg, Path)
    xyzrpy = [pose_msg_to_xyzrpy(p.pose) for p in msg.poses]
    xyzrpy = torch.tensor(xyzrpy)
    return xyzrpy

@timing
def path_length(xyzrpy):
    assert isinstance(xyzrpy, torch.Tensor)
    assert xyzrpy.dim() == 2
    assert xyzrpy.shape[1] == 6
    wp_dists = torch.linalg.norm(xyzrpy[1:, :3] - xyzrpy[:-1, :3], dim=1)
    return wp_dists.sum()

@timing
def xy_to_azimuth(xy):
    """Convert (x, y) coordinates to yaw angle."""
    assert isinstance(xy, torch.Tensor)
    # assert xy.dim() == 2
    assert xy.shape[-1] == 2  # (..., 2)
    # xy_steps = xy[1:, :] - xy[:-1, :]
    # r = xy_steps.norm(dim=1, keepdim=True)
    yaw = torch.atan2(xy[..., 1:], xy[..., :1])
    return yaw


@timing
def xyzrpy_to_path_msg(xyzrpy):
    assert isinstance(xyzrpy, torch.Tensor)
    assert xyzrpy.dim() == 2
    assert xyzrpy.shape[-1] == 6
    xyzrpy = xyzrpy.detach().cpu().numpy()
    msg = Path()
    msg.poses = [PoseStamped(Header(), xyzrpy_to_pose_msg(p)) for p in xyzrpy]
    return msg


def rotation_angle_from_matrix_2d(mat):
    assert isinstance(mat, torch.Tensor)
    assert mat.shape[-2:] == (3, 3)
    # assert (torch.linalg.det(mat) > 0.).all()
    assert (torch.det(mat) > 0.).all()
    # Transform may include scale (e.g. map-to-grid transform).
    if isinstance(mat, np.ndarray):
        angle = np.arctan2(mat[..., 1, 0], mat[..., 0, 0])
    elif isinstance(mat, torch.Tensor):
        angle = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])
    else:
        assert False
        # Assume list of lists.
        angle = math.atan2(mat[1][0], mat[0][0])
    return angle


@timing
def transform_xyyaw_tensor(tf, xyyaw, order=DimOrder.YAW_X_Y):
    """Convert x, y, yaw tensor to grid coordinates."""
    assert isinstance(tf, torch.Tensor)
    assert tf.shape == (3, 3)

    assert isinstance(xyyaw, torch.Tensor)
    assert xyyaw.shape[-1] == 3  # (..., 3)

    assert order in (DimOrder.X_Y_YAW, DimOrder.YAW_X_Y)

    yaw_0 = rotation_angle_from_matrix_2d(tf)

    if order == DimOrder.X_Y_YAW:
        xy = xyyaw[..., :2]
        yaw = xyyaw[..., 2:]
    elif order == DimOrder.YAW_X_Y:
        xy = xyyaw[..., 1:]
        yaw = xyyaw[..., :1]
    # rospy.logdebug('X, Y in map:\n%s', xy)

    xy_grid = transform(tf, xy.transpose(1, 0))
    # rospy.logdebug('X, Y in grid:\n%s', xy_grid)

    # We need to now whether we transform to grid or from grid.
    # yaw_grid = torch.remainder(yaw - yaw_0, 2. * np.pi)

    # TODO: Yaw offset direction?
    # yaw_grid = [yaw - yaw_0 for _, _, yaw in xyyaw]
    # xyyaw_grid = [[xy_grid[0, i].item(), xy_grid[1, i].item(), yaw_grid[i].item()] for i in range(len(yaw_grid))]
    # print('xy_grid shape', xy_grid.shape)
    # print('xyyaw shape', xyyaw.shape)
    if order == DimOrder.X_Y_YAW:
        # xyyaw_out = torch.cat((xy_grid.transpose(1, 0), yaw - yaw_0), dim=1)
        xyyaw_out = torch.cat((xy_grid.transpose(1, 0), yaw + yaw_0), dim=1)
        # Don't transform yaw for now.
        # xyyaw_out = torch.cat((xy_grid.transpose(1, 0), yaw), dim=1)
    elif order == DimOrder.YAW_X_Y:
        # xyyaw_out = torch.cat((yaw - yaw_0, xy_grid.transpose(1, 0)), dim=1)
        xyyaw_out = torch.cat((yaw + yaw_0, xy_grid.transpose(1, 0)), dim=1)
        # Don't transform yaw for now.
        # xyyaw_out = torch.cat((yaw, xy_grid.transpose(1, 0)), dim=1)

    # rospy.loginfo('X, y, yaw in grid: %s', xyyaw_out)
    assert xyyaw_out.shape == (xyyaw.shape[0], 3)

    return xyyaw_out


def log_odds_conversion(rewards, eps = 1e-6):
    assert isinstance(rewards, torch.Tensor)
    assert rewards.dim() == 3
    n_poses, n_cams, n_pts = rewards.shape
    # apply log odds conversion for global voxel map observations update
    p = rewards - rewards.min()
    p = p / p.max()  # normalize rewards to treat them as visibility prob values
    p = torch.clamp(p, 0.5, 1. - eps)  # 0.5 - "unknown", > 0.5 - visible
    lo = torch.log(p / (1. - p))  # compute log odds
    assert lo.shape == (n_poses, n_cams, n_pts)
    lo = lo.view(n_poses * n_cams, n_pts)
    lo_sum = lo.sum(0)
    assert lo_sum.shape == (n_pts,)
    rewards = 1.0 / (1.0 + torch.exp(-lo_sum))  # back to probs from log odds
    return rewards


def reduce_rewards(rewards, eps=1e-6):
    assert isinstance(rewards, torch.Tensor)
    assert rewards.dim() == 3
    n_poses, n_cams, n_pts = rewards.shape
    rewards = torch.clamp(rewards, eps, 1 - eps)
    lo = torch.log(1. - rewards)
    lo = lo.view(n_poses * n_cams, n_pts)
    lo = lo.sum(dim=0)
    rewards = 1. - torch.exp(lo)
    assert rewards.shape == (n_pts,)
    return rewards


def tf_to_pose(tf):
    # tf = Transform()
    pose = Pose()
    pose.position.x = tf.translation.x
    pose.position.y = tf.translation.y
    pose.position.z = tf.translation.z
    pose.orientation = tf.rotation
    return pose


def tf_to_pose_stamped(tf):
    tf = TransformStamped()
    pose = PoseStamped()
    pose.header = tf.header
    pose.pose = tf_to_pose(tf.transform)
    return pose


def isometry_inverse(transform):
    assert isinstance(transform, torch.Tensor)
    assert transform.shape[-2:] == (4, 4)
    inv = torch.zeros(transform.shape, device=transform.device)
    inv[..., :-1, :-1] = transform[..., :-1, :-1].transpose(-1, -2)
    inv[..., :-1, -1:] = -transform[..., :-1, :-1].transpose(-1, -2).matmul(transform[..., :-1, -1:])
    inv[..., -1, -1] = 1.0
    return inv


def rpy_matrix(rpy):
    assert isinstance(rpy, torch.Tensor)
    assert rpy.shape[-1] == 3
    # (N, 3)
    c = torch.cos(rpy)
    s = torch.sin(rpy)
    # c1 = c[..., 0]
    # c2 = c[..., 1]
    # c3 = c[..., 2]
    # s1 = s[..., 0]
    # s2 = s[..., 1]
    # s3 = s[..., 2]
    c1 = c[..., 2]
    c2 = c[..., 1]
    c3 = c[..., 0]
    s1 = s[..., 2]
    s2 = s[..., 1]
    s3 = s[..., 0]
    assert c1.shape == rpy.shape[:-1]
    # rmat = torch.tensor([[c1*c2, c1*s2*s3 - s1*c3, c1*c3*s2 + s1*s3],
    #                      [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3],
    #                      [  -s2,            c2*s3,            c2*c3]])
    rmat = torch.stack((c1*c2, c1*s2*s3 - s1*c3, c1*c3*s2 + s1*s3,
                        s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3,
                          -s2,            c2*s3,            c2*c3), dim=-1)
    # assert rmat.shape == rpy.shape[:-1] + (9,)
    assert rmat.shape[-1] == 9
    # rmat = rmat.reshape((-1, 3, 3))
    rmat = rmat.reshape(rmat.shape[:-1] + (3, 3))
    assert rmat.shape[-2:] == (3, 3)
    # rospy.loginfo('rmat:\n%s', rmat)
    # rospy.loginfo('rmat shape: %s', rmat.shape)
    return rmat


def xyzrpy_matrix(xyzrpy):
    assert isinstance(xyzrpy, torch.Tensor)
    assert xyzrpy.shape[-1] == 6

    # rmat = rpy_matrix(xyzrpy[..., 3:])
    # mat = torch.cat((rmat, xyzrpy[..., :3]), dim=-1)
    # assert mat.shape[-2:] == (3, 4)
    # last_row = torch.zeros(mat.shape[:-2] + (1, 4))
    # last_row[..., 3] = 1.
    # mat = torch.cat((mat, last_row), dim=-2)
    # assert mat.shape[-2:] == (4, 4)
    # assert robot_to_map.shape[-2:] == (4, 4)

    # TODO: Set requires grad?
    mat = torch.zeros(xyzrpy.shape[:-1] + (4, 4), dtype=xyzrpy.dtype, device=xyzrpy.device)
    mat[..., :3, :3] = rpy_matrix(xyzrpy[..., 3:])
    # for i in range(mat.shape[0]):
    #     mat[i] = torch.from_numpy(euler_matrix(axes='sxyz', *xyzrpy[i, 3:].detach().numpy()))
    mat[..., :3, 3] = xyzrpy[..., :3]
    mat[..., 3, 3] = 1.
    assert mat.shape == xyzrpy.shape[:-1] + (4, 4)
    # assert mat.shape[-2:] == (4, 4)

    return mat


def cam_info_fov_planes(msg):
    """Camera info to corner directions and frustum planes."""
    assert isinstance(msg, CameraInfo)
    h, w = msg.height, msg.width
    k_mat = torch.tensor(msg.K, dtype=torch.float32).reshape((3, 3))
    k_mat_inv = torch.inverse(k_mat)
    corners = torch.tensor([[-0.5, w + 0.5, w + 0.5,    -0.5],
                            [-0.5,    -0.5, h + 0.5, h + 0.5],
                            [ 1.0,     1.0,     1.0,     1.0]], dtype=torch.float32)
    n_corners = corners.shape[1]
    corner_dirs = k_mat_inv.matmul(corners)
    assert corner_dirs.shape == (3, 4)
    normals = corner_dirs.roll(1, 1).cross(corner_dirs, dim=0)
    # normals = corner_dirs.cross(corner_dirs.roll(1, 1), dim=0)
    # normals = normals / torch.linalg.norm(normals, 2, dim=0, keepdim=True)
    normals = normals / torch.norm(normals, 2, dim=0, keepdim=True)
    # All planes intersect origin (0, 0, 0).
    planes = torch.cat((normals, torch.zeros((1, n_corners))), dim=0)
    assert corner_dirs.shape == (3, 4)
    assert planes.shape == (4, 4)
    return corner_dirs, planes


@timing
def compute_fov_mask(map, frustums, map_to_cam, ramp_size=1.0):
    """Compute mask indicating which map points are in field of view."""

    assert isinstance(map, torch.Tensor)
    # (4, n_pts)
    assert map.dim() == 2
    assert map.shape[0] == 4
    n_pts = map.shape[1]

    assert isinstance(frustums, torch.Tensor)
    # (n_cams, 4, n_planes)
    assert frustums.dim() == 3
    assert frustums.shape[1] == 4
    n_cams, _, n_planes = frustums.shape
    # assert frustums.shape[0] == map_to_cam.shape[1]

    assert isinstance(map_to_cam, torch.Tensor)
    # (n_poses, n_cams, 4, 4)
    assert map_to_cam.dim() == 4
    n_poses = map_to_cam.shape[0]
    assert map_to_cam.shape[1:] == (n_cams, 4, 4)

    assert ramp_size >= 0.0  # [m]

    # Move frustum planes to map.
    # frustums_in_map = map_to_cam.transpose(-1, -2).matmul(frustums[None])
    # frustums_in_map = frustums[None].transpose(-1, -2).matmul(map_to_cam)
    # assert frustums_in_map.shape == (n_poses, n_cams, 4, n_planes)
    # Compute point to plane and point to frustum distances.
    # plane_dist = frustums_in_map.transpose(-1, -2).matmul(map)
    # assert plane_dist.shape == (n_poses, n_cams, n_planes, n_pts)
    plane_dist = frustums[None].transpose(-1, -2).matmul(map_to_cam).matmul(map)
    assert plane_dist.shape == (n_poses, n_cams, n_planes, n_pts)
    fov_dist, _ = plane_dist.min(dim=-2)
    assert fov_dist.shape == (n_poses, n_cams, n_pts)
    # assert fov_dist.shape == (n_poses, n_cams, n_pts)
    # fov_mask = torch.nn.functional.relu(fov_dist)
    fov_mask = torch.clamp(fov_dist / ramp_size + 1., 0.0, 1.0)
    return fov_mask

@timing
def compute_fov_mask_smooth(map, intrins, map_to_cam, std=10, eps=1e-6):
    t = timer()
    # find points that are observed by the camera (in its FOV)
    assert isinstance(map, torch.Tensor)
    assert isinstance(map_to_cam, torch.Tensor)
    assert isinstance(intrins, dict)
    assert isinstance(intrins['Ks'], torch.Tensor)
    assert isinstance(intrins['hw'], torch.Tensor)
    assert map.dim() == 2
    assert map_to_cam.dim() == 4
    assert intrins['Ks'].dim() == 3
    assert intrins['hw'].dim() == 2
    assert map_to_cam.shape[1] == intrins['Ks'].shape[0]
    assert map_to_cam.shape[1] == intrins['hw'].shape[0]
    assert std > 0.
    n_poses, n_cams = map_to_cam.shape[:2]
    # n_pts = map.shape[1]
    # (n_poses, n_cams, 4, 4) x (4, n_pts)
    pts_cam = map_to_cam.matmul(map)
    # assert pts_cam.shape == (n_poses, n_cams, 4, n_pts)
    # (n_cams, 4, 4) x (n_poses, n_cams, 4, n_pts)
    pts_homo = intrins['Ks'].matmul(pts_cam)
    # assert pts_homo.shape == (n_poses, n_cams, 4, n_pts)
    # depth_constraints = torch.sigmoid(pts_homo[2])
    depth_constraints = pts_homo[:, :, 2, :].sigmoid()
    # assert depth_constraints.shape == (n_poses, n_cams, n_pts)
    # (1, n_cams, 1)
    heights, widths = intrins['hw'][:, 0].reshape([1, n_cams, 1]), intrins['hw'][:, 1].reshape([1, n_cams, 1])
    # assert heights.shape == (1, n_cams, 1)
    width_constraints = torch.exp(
        - std * (pts_homo[:, :, 0, :] / (pts_homo[:, :, 2, :] + eps - widths / 2.) / widths) ** 2)
    # assert width_constraints.shape == (n_poses, n_cams, n_pts)
    height_constraints = torch.exp(
        - std * (pts_homo[:, :, 1, :] / (pts_homo[:, :, 2, :] + eps - heights / 2.) / heights) ** 2)
    # assert height_constraints.shape == (n_poses, n_cams, n_pts)
    fov_mask = depth_constraints * (width_constraints * height_constraints)
    # assert fov_mask.shape == (n_poses, n_cams, n_pts)
    rospy.logdebug('FOV mask computation time: %.3f s', timer() - t)
    return fov_mask


@timing
def compute_vis_mask(map, cam_to_map, param=1.0, voxel_size=0.6):
    assert isinstance(map, torch.Tensor)
    # (4, n_pts)
    assert map.dim() == 2
    assert map.shape[0] == 4
    n_pts = map.shape[1]

    assert isinstance(cam_to_map, torch.Tensor)
    # (n_poses, n_cams, 4, 4)
    assert cam_to_map.dim() == 4
    n_poses, n_cams = cam_to_map.shape[:2]
    assert cam_to_map.shape[1:] == (n_cams, 4, 4)

    assert voxel_size >= 0.0  # [m]

    vis_mask = torch.zeros((n_poses, n_cams, n_pts), device=cam_to_map.device)
    # choose waypoints to compute visibility at
    # based on waypoints distance in the trajectory
    visibilities_dict = OrderedDict()
    for pose_ind in range(n_poses):
        for cam_ind in range(n_cams):
            t_vis1 = timer()
            # if voxel is Free:
            #     voxel = Occupied
            #     compute_visibility
            p = cam_to_map[pose_ind, 0, :3, 3]  # pose of the 0-th camera
            i = int(p[0] / voxel_size)
            j = int(p[1] / voxel_size)
            k = int(p[2] / voxel_size)
            key = str(i) + str(j) + str(k)  # ='ijk'
            if not key in visibilities_dict:
                # compute visibility at selected pose
                origin = cam_to_map[pose_ind, cam_ind, :3, 3:].transpose(-1, -2)
                visibility = point_visibility(map[:3, :].transpose(-1, -2), origin, param=param)
                # store visibility value in a dictionary
                visibilities_dict[key] = visibility
                vis_mask[pose_ind, cam_ind, :] = visibility
                rospy.logdebug('Computing point visibility from pose %i, camera %i at %s: %.3f s',
                               pose_ind, cam_ind, origin.detach().cpu().numpy(), timer() - t_vis1)
            else:
                vis_mask[pose_ind, cam_ind, :] = visibilities_dict[key]
    if len(visibilities_dict) > n_poses:
        visibilities_dict.popitem(last=True)
    assert len(visibilities_dict) <= n_poses
    assert vis_mask.shape == (n_poses, n_cams, n_pts)
    return vis_mask


@timing
def compute_dist_mask(map, cam_to_map, dist_mean=3.0, dist_std=1.5):
    assert isinstance(map, torch.Tensor)
    # (4, n_pts)
    assert map.dim() == 2
    assert map.shape[0] == 4
    n_pts = map.shape[1]

    assert isinstance(cam_to_map, torch.Tensor)
    # (n_poses, n_cams, 4, 4)
    assert cam_to_map.dim() == 4
    n_poses, n_cams = cam_to_map.shape[:2]
    assert cam_to_map.shape[1:] == (n_cams, 4, 4)

    # Distance based mask
    sensor_dist = torch.norm(cam_to_map[..., 3, None] - map, dim=-2)
    assert sensor_dist.shape == (n_poses, n_cams, n_pts)
    dist_mask = torch.exp(-(sensor_dist - dist_mean) ** 2 / dist_std ** 2)
    return dist_mask


class Rewarder(object):

    def __init__(self):
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.robot_frame = rospy.get_param('~robot_frame', 'base_link')
        self.max_age = rospy.get_param('~max_age', 100.0)
        self.debug = rospy.get_param('~debug', True)

        self.fixed_endpoints = rospy.get_param('~fixed_endpoints', ['start'])
        assert isinstance(self.fixed_endpoints, list)

        self.num_cameras = rospy.get_param('~num_cameras', 1)
        self.keep_updating_cameras = rospy.get_param('~keep_updating_cameras', False)
        device_id = rospy.get_param('~gpu_id', 0)
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_id))
            rospy.loginfo("Using GPU device id: %i, name: %s", device_id, torch.cuda.get_device_name(device_id))
        else:
            rospy.loginfo("Using CPU")
            self.device = torch.device("cpu")
        # self.order = DimOrder.YAW_X_Y
        self.order = DimOrder.X_Y_YAW
        assert self.order in (DimOrder.X_Y_YAW, DimOrder.YAW_X_Y)
        self.map_step = rospy.get_param('~map_step', 4)
        self.path_step = rospy.get_param('~path_step', 2)
        self.linear_speed = rospy.get_param('~linear_speed', 1.0)
        self.angular_speed = rospy.get_param('~angular_speed', 1.0)
        self.max_roll = rospy.get_param('~max_roll', 0.7)
        self.max_pitch = rospy.get_param('~max_pitch', 0.7)

        # Latest point cloud map to cover
        self.map_lock = RLock()
        self.map_msg = None
        self.map = None  # n-by-3 cloud position array
        # self.map_x_index = None  # Index of above

        # Latest RPZ manifold
        self.rpz_lock = RLock()
        self.rpz_msg = None
        self.rpz_cloud = None
        self.rpz_all = None
        self.map_to_grid = None
        self.grid_to_map = None

        # Path msg to follow
        self.path_lock = RLock()
        self.path_msg = None
        self.path_xyzrpy = None

        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)

        self.viz_pub = rospy.Publisher('visualization', MarkerArray, queue_size=2)
        self.reward_cloud_pub = rospy.Publisher('~reward_cloud', PointCloud2, queue_size=2)
        self.reward_pub = rospy.Publisher('~reward', Float64, queue_size=1)
        self.publish_reward_cloud = rospy.get_param('~publish_cloud', False)

        # Allow multiple cameras.
        self.cam_info_lock = RLock()
        self.cam_infos = [None] * self.num_cameras
        self.cam_to_robot = [None] * self.num_cameras
        self.cam_corner_dirs = [None] * self.num_cameras
        self.cam_frustums = [None] * self.num_cameras

        self.cam_info_subs = [rospy.Subscriber('camera_info_%i' % i, CameraInfo,
                                               lambda msg, i=i: self.cam_info_received(msg, i), queue_size=2)
                              for i in range(self.num_cameras)]

        self.map_sub = rospy.Subscriber('map', PointCloud2, self.map_received, queue_size=2)
        self.rpz_sub = rospy.Subscriber('rpz', PointCloud2, self.rpz_received, queue_size=2)
        self.path_sub = rospy.Subscriber('path', Path, self.path_received, queue_size=2)
        self.eval_freq = rospy.get_param('~rate', 1.0)
        self.path_timer = rospy.Timer(rospy.Duration(1. / self.eval_freq), self.run)

        # record the metrics
        self.xls_file = os.path.join(rospkg.RosPack().get_path('rpz_planning'), "notebooks/local_rewards_costs.xls")
        self.record_metrics = rospy.get_param('~do_metrics_record', False)
        if self.record_metrics:
            self.wb = xlwt.Workbook()
            self.ws_writer = self.wb.add_sheet("Local rewards and costs")
            self.ws_writer.write(0, 0, 'Time stamp')
            self.ws_writer.write(0, 1, 'Actual reward')
            self.ws_writer.write(0, 2, 'Expected reward')
            self.ws_writer.write(0, 3, 'Actual cost: dist')
            self.ws_writer.write(0, 4, 'Actual cost: turn')
            self.ws_writer.write(0, 5, 'Actual cost: trav')
            self.ws_writer.write(0, 6, 'Actual cost: real_time')
            self.ws_writer.write(0, 7, 'Expected cost: dist')
            self.ws_writer.write(0, 8, 'Expected cost: turn')
            self.ws_writer.write(0, 9, 'Expected cost: trav')
            self.ws_writer.write(0, 10, 'Actual path length')
            self.ws_writer.write(0, 11, 'Expected path length')
            self.row_number = 1

    def lookup_transform(self, target_frame, source_frame, time,
                         no_wait_frame=None, timeout=0.0):

        timeout = rospy.Duration.from_sec(timeout)
        if no_wait_frame is None or no_wait_frame == target_frame:
            tf_s2t = self.tf.lookup_transform(target_frame, source_frame, time, timeout=timeout)
            return tf_s2t

        # Try to get exact transform from no-wait frame to target if available.
        # If not, use most recent transform.
        dont_wait = rospy.Duration.from_sec(0.0)
        try:
            tf_n2t = self.tf.lookup_transform(target_frame, self.odom_frame, time, timeout=dont_wait)
        except tf2_ros.TransformException as ex:
            tf_n2t = self.tf.lookup_transform(target_frame, self.odom_frame, rospy.Time(0))

        # Get the exact transform from source to no-wait frame.
        tf_s2n = self.tf.lookup_transform(self.odom_frame, source_frame, time, timeout=timeout)

        tf_s2t = TransformStamped()
        tf_s2t.header.frame_id = target_frame
        tf_s2t.header.stamp = time
        tf_s2t.child_frame_id = source_frame
        tf_s2t.transform = msgify(Transform,
                                  np.matmul(numpify(tf_n2t.transform),
                                            numpify(tf_s2n.transform)))
        return tf_s2t

    def get_robot_xyzrpy(self, target_frame):
        try:
            transform = self.tf.lookup_transform(target_frame, self.robot_frame,
                                                 rospy.Time.now(), rospy.Duration(3))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            rospy.logwarn('Unable to find robot on the map')
            return None
        tf = transform.transform
        xyzrpy = pose_msg_to_xyzrpy(tf_to_pose(tf))
        return torch.tensor(xyzrpy)

    def visualize_cams(self, pose=None, id=0):
        # TODO: Visualize whole path.
        # TODO: Remove previous indices.
        assert pose is None or isinstance(pose, torch.Tensor)
        # assert pose is None or isinstance(pose, np.ndarray)
        assert id >= 0
        palette = ('r', 'g', 'b', 'c', 'm', 'y')
        # if stamps is None:
        #     stamps = [rospy.Time.now()]
        msg = MarkerArray()
        now = rospy.Time.now()
        with self.cam_info_lock:
            # for k, stamp in enumerate(stamps):
            for i in range(self.num_cameras):
                if self.cam_infos[i] is None:
                    rospy.loginfo('Camera %i not yet available.', i)
                    continue
                # cam_corner_dirs = self.cam_corner_dirs[i]
                # if pose is not None:
                #     cam_corner_dirs = transform(pose.matmul(self.cam_to_robot[i]), cam_corner_dirs)
                rospy.logdebug('Visualizing camera %i...', i)
                marker = Marker()
                # TODO: Allow multiple robot poses.
                dirs = self.cam_corner_dirs[i]
                origin = Point(0., 0., 0.)
                if pose is None:
                    marker.header = self.cam_infos[i].header
                    marker.pose.orientation.w = 1.0
                    # origin = Point(0., 0., 0.)
                else:
                    marker.header.frame_id = self.map_frame
                    # marker.pose.orientation.w = 1.0
                    cam_to_map = pose.matmul(self.cam_to_robot[i].to(pose.device))
                    # cam_to_map = np.matmul(pose, self.cam_to_robot[i])
                    marker.pose = msgify(Pose, cam_to_map.detach().cpu().numpy())
                    # dirs = transform(cam_to_map, dirs)
                    # origin = Point(*cam_to_map[:3, 3].tolist())
                    # origin = Point(0., 0., 0.)

                # marker.stamp = stamp
                marker.header.stamp = now
                marker.action = Marker.MODIFY
                marker.ns = 'fov'
                marker.id = id * self.num_cameras + i
                marker.type = Marker.LINE_LIST
                marker.scale.x = 0.05
                # marker.color.a = 1.0

                c1 = ColorRGBA(*colors.to_rgba(palette[i % len(palette)]))
                c0 = ColorRGBA(*slots(c1))
                c0.a = 0.0

                # From origin to corners.
                marker.points.append(origin)
                marker.colors.append(c0)
                marker.points.append(Point(*dirs[:, 0].tolist()))
                marker.colors.append(c1)

                marker.points.append(origin)
                marker.colors.append(c0)
                marker.points.append(Point(*dirs[:, 1].tolist()))
                marker.colors.append(c1)

                marker.points.append(origin)
                marker.colors.append(c0)
                marker.points.append(Point(*dirs[:, 2].tolist()))
                marker.colors.append(c1)

                marker.points.append(origin)
                marker.colors.append(c0)
                marker.points.append(Point(*dirs[:, 3].tolist()))
                marker.colors.append(c1)

                # Join corners at z = 1.
                marker.points.append(Point(*dirs[:, 0].tolist()))
                marker.colors.append(c1)
                marker.points.append(Point(*dirs[:, 1].tolist()))
                marker.colors.append(c1)

                marker.points.append(Point(*dirs[:, 1].tolist()))
                marker.colors.append(c1)
                marker.points.append(Point(*dirs[:, 2].tolist()))
                marker.colors.append(c1)

                marker.points.append(Point(*dirs[:, 2].tolist()))
                marker.colors.append(c1)
                marker.points.append(Point(*dirs[:, 3].tolist()))
                marker.colors.append(c1)

                marker.points.append(Point(*dirs[:, 3].tolist()))
                marker.colors.append(c1)
                marker.points.append(Point(*dirs[:, 0].tolist()))
                marker.colors.append(c1)

                msg.markers.append(marker)

        self.viz_pub.publish(msg)

    @timing
    def cam_info_received(self, msg, i):
        """Store camera calibration for i-th camera."""
        assert isinstance(msg, CameraInfo)
        assert isinstance(i, int)
        if self.keep_updating_cameras:
            time = rospy.Time.now()
        else:
            time = rospy.Time(0)
        timeout = rospy.Duration.from_sec(1.0)
        try:
            tf = self.tf.lookup_transform(self.robot_frame, msg.header.frame_id, time, timeout)
        except tf2_ros.TransformException as ex:
            rospy.logerr('Could not transform from camera %s to robot %s: %s.',
                         msg.header.frame_id, self.robot_frame, ex)
            return

        tf = torch.tensor(numpify(tf.transform), dtype=torch.float32)
        assert isinstance(tf, torch.Tensor)
        corner_dirs, planes = cam_info_fov_planes(msg)
        with self.cam_info_lock:
            if self.cam_infos[i] is None:
                rospy.loginfo('Got calibration for camera %i (%s).', i, msg.header.frame_id)
            self.cam_infos[i] = msg
            self.cam_to_robot[i] = tf
            self.cam_corner_dirs[i] = corner_dirs
            self.cam_frustums[i] = planes
            self.visualize_cams()
            if not self.keep_updating_cameras:
                self.cam_info_subs[i].unregister()
                rospy.logwarn('Camera %i (%s) unsubscribed.', i, msg.header.frame_id)

    @timing
    def map_received(self, msg):
        """Process and store map for use in reward estimation."""
        t = timer()
        assert isinstance(msg, PointCloud2)

        # This pc transform works slowly (~150 ms).
        # Consider provide input clouds in one coord frame
        if self.map_frame != msg.header.frame_id:
            # Transform the point cloud to global frame
            try:
                transform = self.tf.lookup_transform(self.map_frame, msg.header.frame_id, rospy.Time())
                # msg = do_transform_cloud(msg, transform)
                msg = transform_cloud_msg(numpify(transform.transform), msg)
                msg.header.frame_id = self.map_frame
                rospy.logdebug('Transformed local map to global frame: %s', msg.header.frame_id)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform from %s to %s is not yet available", self.map_frame, msg.header.frame_id)
                return

        rospy.logdebug('Map with %i points received.', msg.height * msg.width)
        map = numpify(msg).ravel()
        np.random.seed(0)  # for reproducibility
        map = np.random.choice(map, map.size // self.map_step)
        # map = np.stack([map[f] for f in ('x', 'y', 'z')])
        # map = np.concatenate([map, np.ones((1, map.shape[-1]))], axis=0)
        map = np.stack([map[f] for f in ('x', 'y', 'z')] + [np.ones((map.size,))])
        map = torch.tensor(map, dtype=torch.float32)
        with self.map_lock:
            self.map_msg = msg
            self.map = map
        rospy.logdebug('Map with %i processed and stored (%.3f s).',
                       map.shape[-1], timer() - t)

    @timing
    def rpz_received(self, msg):
        """Process and store RPZ manifold for use in planning."""
        t = timer()
        assert isinstance(msg, PointCloud2)

        # This pc transform works slowly (~150 ms).
        # Consider providing input clouds in one coord frame
        if self.map_frame != msg.header.frame_id:
            # Transform the point cloud to global frame
            try:
                transform = self.tf.lookup_transform(self.map_frame, msg.header.frame_id, rospy.Time())
                # msg = do_transform_cloud(msg, transform)
                msg = transform_cloud_msg(numpify(transform.transform), msg)
                msg.header.frame_id = self.map_frame
                rospy.loginfo('Transform traversability cloud to global frame: %s',
                              msg.header.frame_id)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Transform from %s to %s is not yet available", self.map_frame, msg.header.frame_id)
                return

        # rospy.loginfo('RPZ: %s (%i, %i)' % (msg.header.frame_id, msg.height, msg.width))
        # sz = (8, 250, 248)
        # assert(msg.height * msg.width == sz[1] * sz[2])
        # cloud = numpify(msg)
        # roll = torch.full(sz, np.nan, dtype=torch.float32)
        # pitch = torch.full(sz, np.nan, dtype=torch.float32)
        # z = torch.full(sz, np.nan, dtype=torch.float32)
        # for i, yaw in enumerate(range(0, 360, 45)):
        #     roll[i, ...] = torch.tensor(cloud['roll_%i' % yaw], dtype=torch.float32).reshape(sz[1:])
        #     pitch[i, ...] = torch.tensor(cloud['pitch_%i' % yaw], dtype=torch.float32).reshape(sz[1:])
        #     z[i, ...] = torch.tensor(cloud['z_%i' % yaw], dtype=torch.float32).reshape(sz[1:])
        # rospy.loginfo(roll.shape)
        # with self.rpz_lock:
        #     self.rpz_msg = msg
        #     self.roll = roll
        #     self.pitch = pitch
        #     self.z = z

        # roll, pitch, z = msg_to_rpz_tensor(msg)
        rpz, cloud = cloud_msg_to_rpz_tensor(msg, self.order)
        cloud_to_grid = cloud_to_grid_transform(cloud)
        grid_to_cloud = torch.inverse(cloud_to_grid)
        rospy.logdebug('Map to grid:\n%s', cloud_to_grid)
        rospy.logdebug('Grid to map:\n%s', grid_to_cloud)
        with self.rpz_lock:
            self.rpz_msg = msg
            # self.roll = roll
            # self.pitch = pitch
            # self.z = z
            self.rpz_cloud = cloud
            # Yaw offset
            # p0 = np.array([self.rpz_cloud['x'][0, 0], self.rpz_cloud['y'][0, 0]])
            # p1 = np.array([self.rpz_cloud['x'][0, 1], self.rpz_cloud['y'][0, 1]])
            # p0 = torch.tensor([self.rpz_cloud['x'][0, 0], self.rpz_cloud['y'][0, 0]])
            # p1 = torch.tensor([self.rpz_cloud['x'][0, 1], self.rpz_cloud['y'][0, 1]])
            # x = (p1 - p0).norm()
            # yaw_offset = torch.atan2(self.rpz_cloud['y'][0, 1] - self.rpz_cloud['y'][0, 0],
            #                          self.rpz_cloud['x'][0, 1] - self.rpz_cloud['x'][0, 0])
            self.rpz_all = rpz
            self.map_to_grid = cloud_to_grid
            self.grid_to_map = grid_to_cloud

        rospy.logdebug('RPZ processed and stored (%.3f s).', (timer() - t))

    @timing
    def get_available_cameras(self):
        """
        Get available cameras:
        camera-to-robot transforms, frustum planes and K-matrixes (intrinsics)
        """
        def K_from_msg(msg):
            k = torch.as_tensor(msg.K).view(3, 3)
            K = torch.eye(4)
            K[:3, :3] = k
            return K

        with self.cam_info_lock:
            cam_to_robot = [tf for tf in self.cam_to_robot if tf is not None]
            if not cam_to_robot:
                return None, None, None
            frustums = [f for f in self.cam_frustums if f is not None]
            intrins = {'Ks': [K_from_msg(msg) for msg in self.cam_infos if msg is not None],
                       'hw': [torch.tensor([msg.height, msg.width]) for msg in self.cam_infos if msg is not None]}
            assert len(cam_to_robot) == len(frustums)

        cam_to_robot = torch.stack(cam_to_robot)
        # n_cams, 4, 4
        assert cam_to_robot.dim() == 3
        assert cam_to_robot.shape[1:] == (4, 4)

        frustums = torch.stack(frustums)
        # n_cams, 4, n_planes
        assert frustums.dim() == 3
        assert frustums.shape[-2] == 4

        intrins['Ks'] = torch.stack(intrins['Ks'])
        # n_cams, 4, 4
        assert intrins['Ks'].dim() == 3
        assert intrins['Ks'].shape[-2] == 4

        intrins['hw'] = torch.stack(intrins['hw'])
        # n_cams, 2
        assert intrins['hw'].dim() == 2
        assert intrins['hw'].shape[-1] == 2

        return cam_to_robot, frustums, intrins

    def path_reward(self, xyzrpy, map, cam_to_robot, frustums, vis_cams=False):
        assert isinstance(xyzrpy, torch.Tensor)
        # (..., N, 6)
        assert xyzrpy.shape[-1] == 6
        assert xyzrpy.dim() >= 2

        # n_cams, 4, n_planes
        n_cams, _, n_planes = frustums.shape
        assert cam_to_robot.shape == (n_cams, 4, 4)
        assert xyzrpy.shape[1] == 6
        n_poses = xyzrpy.shape[0]

        t = timer()
        cam_to_robot = cam_to_robot.to(self.device)
        frustums = frustums.to(self.device)
        map = map.to(self.device)
        assert map.shape[0] == 4
        n_pts = map.shape[-1]
        xyzrpy = xyzrpy.to(self.device)
        rospy.logdebug('Moving to %s: %.3f s', self.device, timer() - t)

        # Keep map coordinates, convert to grid just for RPZ interpolation.
        # Assume map-to-grid transform being 2D similarity with optional z offset.
        # Optimize xy pairs, with yaw defined by xy steps.
        # Allow start and/or end xy fixed.
        # For interpolation we'll have: rpz(to_grid(xyyaw)).
        # Optionally, offset z if needed.

        # Prepare reward cloud for visualization.
        reward_cloud = np.zeros((n_pts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                 ('visibility', 'f4'), ('fov', 'f4'), ('distance', 'f4'),
                                                 ('reward', 'f4')])

        for i, f in enumerate(['x', 'y', 'z']):
            reward_cloud[f] = map[i].detach().cpu().numpy()

        # Get camera to map transforms.
        t = timer()
        robot_to_map = xyzrpy_matrix(xyzrpy)
        assert robot_to_map.shape == (n_poses, 4, 4)
        cam_to_map = robot_to_map[:, None].matmul(cam_to_robot[None])
        assert cam_to_map.shape == (n_poses, n_cams, 4, 4)
        map_to_cam = isometry_inverse(cam_to_map)
        assert map_to_cam.shape == (n_poses, n_cams, 4, 4)
        rospy.logdebug('Camera to map transforms: %.3f s', timer() - t)

        # Visibility / occlusion mask.
        t = timer()
        vis_mask = compute_vis_mask(map, cam_to_map, param=0.01)
        with torch.no_grad():
            reward_cloud['visibility'] = reduce_rewards(vis_mask).detach().cpu().numpy()
        rospy.logdebug('Point visibility computation took: %.3f s.', timer() - t)

        # compute smooth version of FOV mask
        # fov_mask = compute_fov_mask_smooth(map, intrins, map_to_cam)
        fov_mask = compute_fov_mask(map, frustums, map_to_cam)
        assert fov_mask.shape == (n_poses, n_cams, n_pts)
        with torch.no_grad():
            reward_cloud['fov'] = reduce_rewards(fov_mask).detach().cpu().numpy()

        # Compute point to sensor distances.
        # TODO: Optimize, avoid exhaustive distance computation.
        dist_mask = compute_dist_mask(map, cam_to_map)
        with torch.no_grad():
            reward_cloud['distance'] = reduce_rewards(dist_mask).detach().cpu().numpy()

        # Compute rewards
        rewards = vis_mask * fov_mask * dist_mask
        assert rewards.shape == (n_poses, n_cams, n_pts)

        # share and sum rewards over multiple sensors and view points
        # rewards = log_odds_conversion(rewards)
        # instead of log odds: max of rewards over all sensors and wps poses
        rewards = reduce_rewards(rewards)

        reward = rewards.sum()
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == ()

        # reward cloud for visualization
        reward_cloud['reward'] = rewards.detach().cpu().numpy()

        # Visualize cameras (first, middle and last).
        if vis_cams:
            t = timer()
            self.visualize_cams(robot_to_map[0].detach(), id=0)
            self.visualize_cams(robot_to_map[n_poses // 2].detach(), id=1)
            self.visualize_cams(robot_to_map[-1].detach(), id=2)
            rospy.logdebug('Cameras visualized for %i poses (%.3f s).', 3, timer() - t)

        return reward, reward_cloud

    def path_cost(self, xyzrpy):
        assert isinstance(xyzrpy, torch.Tensor)
        # (..., N, 6)
        assert xyzrpy.shape[-1] == 6
        assert xyzrpy.dim() >= 2

        # Distance cost.
        # xy_diff = torch.diff(xyzrpy[..., :2], dim=-2)
        xy_diff = xyzrpy[..., 1:, :2] - xyzrpy[..., :-1, :2]
        edges = xy_diff.norm(dim=-1, keepdim=True)
        dist_cost = edges.sum() / self.linear_speed
        # rospy.logdebug('Distance cost: %.1f s.', dist_cost.item())

        # Turning cost.
        # yaw_diff = torch.diff(xyzrpy[..., -1:], dim=-2).abs()
        yaw_diff = (xyzrpy[..., 1:, -1:] - xyzrpy[..., :-1, -1:]).abs()
        # yaw_diff = torch.remainder(torch.diff(xyzrpy[..., -1:], dim=-2), 2 * np.pi)
        yaw_diff = torch.min(yaw_diff, 2 * np.pi - yaw_diff)
        turn_cost = yaw_diff.sum() / self.angular_speed
        # rospy.logdebug('Turning cost: %.1f s.', turn_cost.item())

        # Traversability cost, penalty for roll and pitch.
        # TODO: Convert to time cost using control parameters.
        rp = xyzrpy[..., 1:, 3:5]
        # Use edge lenghts to scale roll and pitch penalties.
        rp = rp * edges
        # trav_cost = rp.abs().mean()
        trav_cost = (rp.abs().sum(dim=0) / torch.tensor([self.max_roll, self.max_pitch]).to(xyzrpy.device)).sum()

        return dist_cost, turn_cost, trav_cost

    @timing
    def path_received(self, msg):
        assert isinstance(msg, Path)
        # # Discard old messages.
        # age = (rospy.Time.now() - msg.header.stamp).to_sec()
        # if age > self.max_age:
        #     rospy.logwarn('Discarding path %.1f s > %.1f s old.', age, self.max_age)
        #     return

        # Subsample input path to reduce computation.
        if self.path_step > 1:
            msg.poses = msg.poses[::self.path_step]

        # Check compatibility of path and map frames.
        if not msg.header.frame_id:
            rospy.logwarn_once('Map frame %s will be used instead of empty path frame.',
                               self.map_frame)
            msg.header.frame_id = self.map_frame
        elif not self.map_frame:
            self.map_frame = msg.header.frame_id
        elif self.map_frame and msg.header.frame_id != self.map_frame:
            rospy.logwarn_once('Map frame %s will be used instead of path frame %s.',
                               self.map_frame, msg.header.frame_id)
        with self.path_lock:
            self.path_msg = msg
            self.path_xyzrpy = path_msg_to_xyzrpy(self.path_msg)

        # Get RPZ subspace and map for optimization.
        with self.rpz_lock:
            if self.rpz_msg is None:
                rospy.logwarn('Skipping path. RPZ cloud not yet received.')
                return
            assert isinstance(self.rpz_msg, PointCloud2)
            assert self.rpz_msg.header.frame_id == self.path_msg.header.frame_id
            rpz_all = self.rpz_all
            map_to_grid = self.map_to_grid
            grid_to_map = self.grid_to_map
        assert map_to_grid.shape == (3, 3)
        assert grid_to_map.shape == (3, 3)
        # rospy.loginfo('Map to grid:\n%s', map_to_grid.detach().numpy())
        rospy.logdebug('Grid to map:\n%s', grid_to_map.detach().numpy())

        assert self.path_xyzrpy.dim() == 2
        assert self.path_xyzrpy.shape[1] == 6
        n_poses = self.path_xyzrpy.shape[0]

        # yaw = xy_to_azimuth(xy)
        xy = self.path_xyzrpy[:, :2]
        yaw_tail = xy_to_azimuth(xy[1:, :] - xy[:-1, :])
        # TODO: Add starting yaw.
        yaw = torch.cat((yaw_tail[:1, :], yaw_tail), dim=-2)
        # rospy.loginfo('Yaw:\n%s', np.degrees(yaw.detach().numpy()))
        if self.order == DimOrder.X_Y_YAW:
            xyyaw = torch.cat((xy, yaw), dim=-1)
        elif self.order == DimOrder.YAW_X_Y:
            xyyaw = torch.cat((yaw, xy), dim=-1)

        assert xyyaw.shape == (n_poses, 3)
        xyyaw_grid = transform_xyyaw_tensor(map_to_grid, xyyaw, order=self.order)
        rpz = interpolate_rpz(rpz_all, xyyaw_grid, order=self.order)

        # Transform z?
        assert rpz.shape == (n_poses, 3)
        assert xyyaw.shape == (n_poses, 3)
        if self.order == DimOrder.X_Y_YAW:
            # Fix yaw from XY.
            self.path_xyzrpy = torch.cat((xyyaw[:, :2], rpz[:, 2:], rpz[:, :2], xyyaw[:, 2:]), dim=-1)
        elif self.order == DimOrder.YAW_X_Y:
            self.path_xyzrpy = torch.cat((xyyaw[:, 1:], rpz[:, 2:], rpz[:, :2], xyyaw[:, :1]), dim=-1)
        assert self.path_xyzrpy.dim() == 2
        assert self.path_xyzrpy.shape[1] == 6

    def run(self, event):
        with self.path_lock:
            if self.path_msg is None:
                rospy.logwarn('Path is not yet received.')
                return

        with self.map_lock:
            if self.map_msg is None:
                rospy.logwarn('Map cloud is not yet received.')
                return
            assert isinstance(self.map_msg, PointCloud2)
            assert self.map_msg.header.frame_id == self.path_msg.header.frame_id

        # Get frustums and intrinsics of available cameras.
        cam_to_robot, frustums, _ = self.get_available_cameras()
        if cam_to_robot is None:
            rospy.logwarn('No cameras available.')
            return

        map = self.map
        path_xyzrpy = self.path_xyzrpy
        path_msg = self.path_msg

        if torch.isnan(path_xyzrpy).any():
            rospy.logwarn("Path contains NANs. Evaluation will not be performed this time")
            return

        xyzrpy = self.get_robot_xyzrpy(path_msg.header.frame_id)

        if xyzrpy is None:
            rospy.logwarn('Unable to find robot pose on the map')
            return

        # construct actual path with ~the same dists between waypoints
        prev_pose = xyzrpy[:3]
        actual_path_xyzrpy = [xyzrpy.unsqueeze(0)]
        wp_dists = torch.linalg.norm(path_xyzrpy[1:, :3] - path_xyzrpy[:-1, :3], dim=1)
        received_new_path = False
        traj_time = 0.0
        while not received_new_path:
            traj_time = (self.path_msg.header.stamp - path_msg.header.stamp).to_sec()
            received_new_path = traj_time > 0
            xyzrpy = self.get_robot_xyzrpy(target_frame=self.path_msg.header.frame_id)

            if xyzrpy is not None:
                cur_pose = xyzrpy[:3]
                dp = torch.linalg.norm(cur_pose - prev_pose)
                if dp > wp_dists.min():
                    actual_path_xyzrpy.append(xyzrpy.unsqueeze(0))
                    prev_pose = cur_pose

                    if len(actual_path_xyzrpy) == path_xyzrpy.shape[0] or \
                            path_length(torch.cat(actual_path_xyzrpy, dim=0)) >= path_length(path_xyzrpy):
                        break
        # xyzrpy = self.get_robot_xyzrpy(target_frame=self.path_msg.header.frame_id)
        # if xyzrpy is not None:
        #     actual_path_xyzrpy.append(xyzrpy[:3].unsqueeze(0))

        actual_path_xyzrpy = torch.cat(actual_path_xyzrpy, dim=0)
        path_xyzrpy = path_xyzrpy[:actual_path_xyzrpy.shape[0], :]

        # publish actual path
        assert actual_path_xyzrpy.shape[0] <= path_xyzrpy.shape[0]
        assert actual_path_xyzrpy.shape[1] == path_xyzrpy.shape[1]  # (N1, 6), (N2, 6), N1 <= N2
        actual_path_msg = xyzrpy_to_path_msg(actual_path_xyzrpy)
        actual_path_msg.header = self.path_msg.header
        path_pub = rospy.Publisher('~actual_path', Path, queue_size=1)
        path_pub.publish(actual_path_msg)

        # compute rewards from path and point cloud map
        t0 = timer()
        # rewards
        expected_reward, _ = self.path_reward(path_xyzrpy, map, cam_to_robot, frustums, vis_cams=False)
        actual_reward, actual_reward_cloud = self.path_reward(actual_path_xyzrpy, map, cam_to_robot, frustums, vis_cams=True)
        # costs
        expected_cost = {}
        expected_cost['dist'], expected_cost['turn'], expected_cost['trav'] = self.path_cost(path_xyzrpy)
        actual_cost = {}
        actual_cost['dist'], actual_cost['turn'], actual_cost['trav'] = self.path_cost(actual_path_xyzrpy)
        actual_cost['real_time'] = traj_time
        # lengths
        expected_length = path_length(path_xyzrpy)
        actual_length = path_length(actual_path_xyzrpy)
        dt = timer() - t0

        rospy.logdebug('N points in map: %i, compute time: %.3f sec', map.shape[-1], dt)
        rospy.loginfo('Expected Path length: %.1f, N wps: %i, '
                      'reward: %.1f, cost: %.1f',
                      expected_length, path_xyzrpy.shape[0],
                      expected_reward, expected_cost['dist']+expected_cost['turn']+expected_cost['trav'])
        rospy.loginfo('Actual Path length: %.1f, N wps: %i, '
                      'reward: %.1f, cost: %.1f',
                      actual_length, actual_path_xyzrpy.shape[0],
                      actual_reward, actual_cost['dist']+actual_cost['turn']+actual_cost['trav'])

        # record data
        if self.record_metrics:
            self.ws_writer.write(self.row_number, 0, self.path_msg.header.stamp.to_sec())
            self.ws_writer.write(self.row_number, 1, f'{actual_reward}')
            self.ws_writer.write(self.row_number, 2, f'{expected_reward}')
            self.ws_writer.write(self.row_number, 3, f'{actual_cost["dist"]}')
            self.ws_writer.write(self.row_number, 4, f'{actual_cost["turn"]}')
            self.ws_writer.write(self.row_number, 5, f'{actual_cost["trav"]}')
            self.ws_writer.write(self.row_number, 6, f'{actual_cost["real_time"]}')
            self.ws_writer.write(self.row_number, 7, f'{expected_cost["dist"]}')
            self.ws_writer.write(self.row_number, 8, f'{expected_cost["turn"]}')
            self.ws_writer.write(self.row_number, 9, f'{expected_cost["trav"]}')
            self.ws_writer.write(self.row_number, 10, f'{actual_length}')
            self.ws_writer.write(self.row_number, 11, f'{expected_length}')
            self.row_number += 1
            self.wb.save(self.xls_file)

        # publish reward value and cloud
        if expected_reward is not None:
            self.reward_pub.publish(Float64(expected_reward.item()))
            if self.publish_reward_cloud:
                reward_cloud_msg = msgify(PointCloud2, actual_reward_cloud)
                assert isinstance(reward_cloud_msg, PointCloud2)
                reward_cloud_msg.header = self.path_msg.header
                self.reward_cloud_pub.publish(reward_cloud_msg)


if __name__ == '__main__':
    rospy.init_node('local_actual_rewarder', log_level=rospy.INFO)
    node = Rewarder()
    rospy.spin()
