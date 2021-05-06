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
