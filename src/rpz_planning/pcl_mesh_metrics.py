#!/usr/bin/env python

from typing import Union
import torch

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d._C import face_point_dist_forward as face_point_distance
from pytorch3d._C import edge_point_dist_forward as edge_point_distance
from pytorch3d._C import point_face_dist_forward as point_face_distance
from pytorch3d._C import point_edge_dist_forward as point_edge_distance
from pytorch3d.ops.knn import knn_points


def face_point_distance_truncated(meshes: Meshes, pcls: Pointclouds):
    # TODO: potential bug
    """
    `face_point_distance_truncated(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # face to point distance: shape (T,)
    face_to_point = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris
    )[0].sqrt()

    # weight each example by the inverse of number of faces in the example
    tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
    num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
    weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
    weights_t = 1.0 / weights_t.float()
    face_to_point = face_to_point * weights_t
    face_dist = face_to_point.sum() / N

    return face_dist


def point_face_distance_truncated(meshes: Meshes, pcls: Pointclouds):
    """
    `point_face_distance_truncated(mesh, pcl)`:
    Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl

    The above distance function is applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )[0].sqrt()

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / N

    return point_dist


def edge_point_distance_truncated(meshes: Meshes, pcls: Pointclouds):
    # TODO: potential bug
    """
    `edge_point_distance_truncated(mesh, pcl)`: Computes the squared distance of each edge segment in mesh
        to the closest point in pcl and averages across all edges in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `edge_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()

    # packed representation for edges
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.edges_packed()
    segms = verts_packed[edges_packed]  # (S, 2, 3)
    segms_first_idx = meshes.mesh_to_edges_packed_first_idx()
    max_segms = meshes.num_edges_per_mesh().max().item()

    # edge to edge distance: shape (S,)
    edge_to_point = edge_point_distance(
        points, points_first_idx, segms, segms_first_idx, max_segms
    )[0].sqrt()

    # weight each example by the inverse of number of edges in the example
    segm_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(S_n),)
    num_segms_per_mesh = meshes.num_edges_per_mesh()  # (N,)
    weights_s = num_segms_per_mesh.gather(0, segm_to_mesh_idx)
    weights_s = 1.0 / weights_s.float()
    edge_to_point = edge_to_point * weights_s
    edge_dist = edge_to_point.sum() / N

    return edge_dist


def point_edge_distance_truncated(meshes: Meshes, pcls: Pointclouds):
    """
    `point_edge_distance_truncated(mesh, pcl)`:
    Computes the squared distance of each point p in pcl
        to the closest edge segment in mesh and averages across all points in pcl

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_edge(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """
    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for edges
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.edges_packed()
    segms = verts_packed[edges_packed]  # (S, 2, 3)
    segms_first_idx = meshes.mesh_to_edges_packed_first_idx()

    # point to edge distance: shape (P,)
    point_to_edge = point_edge_distance(
        points, points_first_idx, segms, segms_first_idx, max_points
    )[0].sqrt()

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = pcls.packed_to_cloud_idx()  # (sum(P_i), )
    num_points_per_cloud = pcls.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_edge = point_to_edge * weights_p
    point_dist = point_to_edge.sum() / N

    return point_dist


def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths


def chamfer_distance_truncated(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    apply_point_reduction=True,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        apply_point_reduction: Whether to apply points reduction. If set to True returns
            one distance per batch (of shape (N, 1)). Otherwise, result size is equal to (N, P1).
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
    """
    x, x_lengths = _handle_pointcloud_input(x, x_lengths)
    y, y_lengths = _handle_pointcloud_input(y, y_lengths)

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)

    # knn_points returns squared distances
    cham_x = x_nn.dists[..., 0].sqrt()  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if apply_point_reduction:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)

        if point_reduction == "mean":
            cham_x /= x_lengths

        if batch_reduction is not None:
            # batch_reduction == "sum"
            cham_x = cham_x.sum()

            if batch_reduction == "mean":
                cham_x /= N

    return cham_x
