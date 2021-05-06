from __future__ import absolute_import, division, print_function
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
from scipy.spatial import ConvexHull
import torch


def point_visibility(pts, origin, radius=None, param=3.0):
    assert isinstance(pts, torch.Tensor)
    assert isinstance(origin, torch.Tensor)
    assert pts.device == origin.device
    assert pts.dim() == origin.dim()
    assert pts.shape[-1] == origin.shape[-1]
    dirs = pts - origin
    dist = torch.norm(dirs, dim=-1, keepdim=True)
    assert dist.shape == (pts.shape[0], 1)
    if radius is None:
        radius = dist.max() * 10.**param
    # Mirror points behind the sphere.
    pts_flipped = pts + 2.0 * (radius - dist) * (dirs / dist)
    # TODO: Allow flexible leading dimensions (convhull needs (npts, ndims)).
    conv_hull = ConvexHull(pts_flipped.detach().cpu().numpy())
    # TODO: Use distance from convex hull to smooth the indicator?
    mask = torch.zeros((pts.shape[0],), device=pts.device)
    mask[conv_hull.vertices] = 1.
    return mask
