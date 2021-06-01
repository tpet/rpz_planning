#!/usr/bin/env python

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d._C import face_point_dist_forward as face_point_distance
from pytorch3d._C import edge_point_dist_forward as edge_point_distance


def face_point_distance_weighted(meshes: Meshes, pcls: Pointclouds):
    """
    `face_point_distance_weighted(mesh, pcl)`: Computes the squared distance of each triangular face in
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
    face_to_point, _ = face_point_distance(
        points, points_first_idx, tris, tris_first_idx, max_tris
    )

    # weight each example by the inverse of number of faces in the example
    tri_to_mesh_idx = meshes.faces_packed_to_mesh_idx()  # (sum(T_n),)
    num_tris_per_mesh = meshes.num_faces_per_mesh()  # (N, )
    weights_t = num_tris_per_mesh.gather(0, tri_to_mesh_idx)
    weights_t = 1.0 / weights_t.float()
    face_to_point = face_to_point * weights_t
    face_dist = face_to_point.sum() / N

    return face_dist


def edge_point_distance_weighted(meshes: Meshes, pcls: Pointclouds):
    """
    `edge_point_distance_weighted(mesh, pcl)`: Computes the squared distance of each edge segment in mesh
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
    edge_to_point, _ = edge_point_distance(
        points, points_first_idx, segms, segms_first_idx, max_segms
    )

    # weight each example by the inverse of number of edges in the example
    segm_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(S_n),)
    num_segms_per_mesh = meshes.num_edges_per_mesh()  # (N,)
    weights_s = num_segms_per_mesh.gather(0, segm_to_mesh_idx)
    weights_s = 1.0 / weights_s.float()
    edge_to_point = edge_to_point * weights_s
    edge_dist = edge_to_point.sum() / N

    return edge_dist
