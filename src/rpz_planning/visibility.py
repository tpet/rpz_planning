from __future__ import absolute_import, division, print_function
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
from scipy.spatial import ConvexHull
import torch


def point_visibility(pts, origin, radius=None):
    assert isinstance(pts, torch.Tensor)
    assert isinstance(origin, torch.Tensor)
    assert pts.dim() == origin.dim()
    assert pts.shape[-1] == origin.shape[-1]
    # assert pts.shape[-1] == 3
    # assert origin[1].shape[-1] == 3
    dir = pts - origin
    dist = torch.norm(dir, dim=-1, keepdim=True)
    if radius is None:
        radius = torch.max(dist) * 10.**3.
    # Mirror points behind the sphere.
    pts_flipped = (1. + 2. * (radius - dist)) * pts
    conv_hull = ConvexHull(pts_flipped.detach().numpy())
    # TODO: Use distance from convex hull to smooth the indicator?
    mask = torch.zeros((pts.shape[0]))
    mask[conv_hull.vertices] = 1.
    return mask


# Torch HPR
def sphericalFlip(points, device, param):
    assert isinstance(points, torch.Tensor)
    assert points.dim() == 2
    assert points.shape[1] == 3
    """
    Function used to Perform Spherical Flip on the Original Point Cloud
    """
    n = len(points)  # total number of points
    normPoints = torch.norm(points, dim=1)  # Normed points, sqrt(x^2 + y^2 + z^2)

    radius = torch.max(normPoints) * 10.0**param  # Radius of a sphere
    flippedPointsTemp = 2 * torch.repeat_interleave((radius - normPoints).view(n, 1), len(points[0]), dim=1) * points
    # Apply Equation to get Flipped Points
    flippedPoints = flippedPointsTemp / torch.repeat_interleave(normPoints.view(n, 1), len(points[0]), dim=1).to(device)

    flippedPoints += points
    return flippedPoints


def convexHull(points, device):
    """
    Function used to Obtain the Convex hull
    """
    points = torch.cat([points, torch.zeros((1, 3), device=device)], dim=0)  # All points plus origin
    hull = ConvexHull(points.cpu().numpy())  # Visible points plus possible origin. Use its vertices property.
    return hull


def hidden_pts_removal(points, R_param=2):
    """
    :param points: input point cloud, Nx3
    :type points: torch.Tensor
    :param device: CPU, torch.device("cpu"), or GPU, torch.device("cuda")
    :returns: point cloud after HPR algorithm, Nx3
    :rtype: torch.Tensor
    """
    assert isinstance(points, torch.Tensor)
    assert points.dim() == 2
    assert points.shape[1] == 3
    assert R_param > 0
    # Initialize the points visible from camera location
    flippedPoints = sphericalFlip(points, points.device, R_param)

    visibleHull = convexHull(flippedPoints, points.device)
    visibleVertex = visibleHull.vertices[:-1]  # indexes of visible points
    # convert indexes to mask
    visibleMask = torch.zeros(points.size()[0], device=points.device)
    visibleMask[visibleVertex] = 1

    pts_visible = points[visibleVertex, :]
    return pts_visible, visibleMask
