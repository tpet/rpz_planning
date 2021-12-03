import xml.etree.ElementTree as ET
import collada, time, numpy as np, uuid, sys
from math import pi
import json, requests
import os
import zipfile
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy


def pose_to_matrix(pose):
    assert len(pose) == 6
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', pose[3:], degrees=False).as_matrix()
    T[:3, 3] = pose[:3]
    return T


def find_character_in_string(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def get_sdf_path(model_path, model_name):
    sdf_path = os.path.join(
        model_path + model_name,
        os.listdir(model_path + model_name)[0],
        'model.sdf')
    return sdf_path


def generate_geometry(mesh, box, pose):
    id = str(uuid.uuid4())
    vert = np.array([0,1,1, 1,1,1, 0,0,1, 1,0,1,
                     0,1,0, 1,1,0, 0,0,0, 1,0,0], dtype='float32').reshape(-1, 3) - .5
    vert[:, 0] = vert[:, 0] * box[0] + pose[0]
    vert[:, 1] = vert[:, 1] * box[1] + pose[1]
    vert[:, 2] = vert[:, 2] * box[2] + pose[2]
    vert = vert.reshape(-1)

    normal = np.array([0,0,1,  0,0,1,  0,0,1,  0,0,1,
                       0,1,0,  0,1,0,  0,1,0,  0,1,0,
                       0,-1,0, 0,-1,0, 0,-1,0, 0,-1,0,
                       -1,0,0, -1,0,0, -1,0,0, -1,0,0,
                       1,0,0,  1,0,0,  1,0,0,  1,0,0,
                       0,0,-1, 0,0,-1, 0,0,-1, 0,0,-1])
    vert_src = collada.source.FloatSource('vsrc' + id, vert, ('X', 'Y', 'Z'))
    normal_src = collada.source.FloatSource('nsrc' + id, normal, ('X', 'Y', 'Z'))

    geom = collada.geometry.Geometry(mesh, 'geometry' + id, 'mygeom', [vert_src, normal_src])

    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', '#vsrc' + id)
    input_list.addInput(1, 'NORMAL', '#nsrc' + id)
    indices = np.array([0,0,2,1,3,2,0,0,3,2,1,3,0,4,1,5,5,6,0,
                           4,5,6,4,7,6,8,7,9,3,10,6,8,3,10,2,11,0,12,
                           4,13,6,14,0,12,6,14,2,15,3,16,7,17,5,18,3,
                           16,5,18,1,19,5,20,7,21,6,22,5,20,6,22,4,23])
    triset = geom.createTriangleSet(indices, input_list, 'mat')
    geom.primitives.append(triset)
    return geom, id


def generate_ground_plane():
    mesh = collada.Collada()
    geom, id = generate_geometry(mesh, [100, 100, .1], [0, 0, .05, 0, 0, 0])
    mesh.geometries.append(geom)
    geomnode = collada.scene.GeometryNode(geom)
    node = collada.scene.Node('node' + id, children=[geomnode])
    scene = collada.scene.Scene('scene' + id, [node])
    mesh.scenes.append(scene)
    mesh.scene = scene
    return [mesh]


def parse_sdf(model_name):
    if model_name == 'tunnel_staging_area':
        model_name = 'subt_tunnel_staging_area'
    elif model_name == 'Extinguisher':
        model_name = 'Fire Extinguisher'

    root_sdf = None
    for model_path in models_path:
        try:
            model_name = model_name.lower()
            sdf_path = get_sdf_path(model_path, model_name)
            root_sdf = ET.parse(sdf_path).getroot()
            break
        except FileNotFoundError:
            try:
                model_name = model_name.lower().replace(' ', '_')
                sdf_path = get_sdf_path(model_path, model_name)
                root_sdf = ET.parse(sdf_path).getroot()
                break
            except FileNotFoundError:
                try:
                    model_name = model_name.lower().replace('_', ' ')
                    sdf_path = get_sdf_path(model_path, model_name)
                    root_sdf = ET.parse(sdf_path).getroot()
                    break
                except:
                    # https://github.com/ignitionrobotics/ign-fuel-tools
                    print('Use ign-fuel-tools to download model: ', model_name)
                    print('For example, execute: \
                        ign fuel download -u  https://fuel.ignitionrobotics.org/1.0/OpenRobotics/models/Large\ Rock\ Fall -v 4')
                    continue

    if root_sdf is None:
        return []
    root_pose = root_sdf.find('model/link/pose').text if root_sdf.find('model/link/pose') is not None else '0 0 0 0 0 0'
    root_pose = [float(a) for a in filter(lambda a: bool(a), root_pose.split(' '))]
    T = pose_to_matrix(root_pose)

    meshes = []
    for node in root_sdf.findall('model/link/collision'):
        pose = node.find('pose').text if node.find('pose') is not None else '0 0 0 0 0 0'
        pose = [float(a) for a in filter(lambda a: bool(a), pose.split(' '))]
        T = T @ pose_to_matrix(pose)

        if node.find('geometry/mesh') is None:
            # box = node.find('geometry/box/size').text
            # box = [float(a) for a in filter(lambda a: bool(a), box.split(' '))]
            #
            # mesh = collada.Collada()
            # geom, id = generate_geometry(mesh, box, pose)
            #
            # mesh.geometries.append(geom)
            # geomnode = collada.scene.GeometryNode(geom)
            # node = collada.scene.Node('node' + id, children=[geomnode])
            # scene = collada.scene.Scene('scene' + id, [node])
            # mesh.scenes.append(scene)
            # mesh.scene = scene
            # meshes.append(mesh)
            pass
        else:  # Importing DAE collision mesh
            # print('DAE detected !')
            dae_file = node.find('geometry/mesh/uri').text
            if 'https://' in dae_file:
                dae_file = dae_file[find_character_in_string(dae_file, '/')[-2] + 1:]

            dae_path = os.path.join(
                os.path.join(model_path, model_name),
                os.listdir(model_path + model_name)[0],
                dae_file)

            scale = node.find('geometry/mesh/scale').text \
                if node.find('geometry/mesh/scale') is not None else '1 1 1'
            scale = [float(a) for a in filter(lambda a: bool(a), scale.split(' '))]

            mesh = collada.Collada(dae_path)

            nodes = []
            for child in mesh.scene.nodes:
                if len(child.transforms) != 0 and isinstance(child.transforms[0], collada.scene.MatrixTransform):
                    T = T @ child.transforms[0].matrix
                    if (len(child.transforms)) > 1:
                        scale = np.asarray(scale) * np.asarray(
                            [child.transforms.x, child.transforms.y, child.transforms.z])
                    child.transforms = []
                    child.transforms.append(collada.scene.MatrixTransform(T.flatten()))
                    parent = collada.scene.Node('parent_' + child.id, children=[child])
                else:
                    parent = child
                parent.transforms.append(collada.scene.ScaleTransform(scale[0], scale[1], scale[2]))
                nodes.append(parent)
            mesh.scene.nodes = nodes
            meshes.append(mesh)

    return meshes


def parse_model(model_name):
    extracted_meshes = parse_sdf(model_name)
    pose = [0, 0, 0, 0, 0, 0]
    scale = [1, 1, 1]
    meshes = []
    
    for mesh in extracted_meshes:
        for i, node in enumerate(mesh.scene.nodes):
            if node.id is not None:
                parent = collada.scene.Node('parent__' + node.id, children=[node])

                T = pose_to_matrix(pose)

                parent.transforms.append(collada.scene.MatrixTransform(T.flatten()))
                parent.transforms.append(collada.scene.ScaleTransform(scale[0], scale[1], scale[2]))
                mesh.scene.nodes[i] = parent
            else:
                print("Couldn't find transform between parent and child nodes:", model_name)

    meshes.extend(extracted_meshes)

    # meshes.extend(generate_ground_plane())
    # output = merge_meshes(meshes)
    # output.write(f'./meshes/{sys.argv[2]}.dae')
    return merge_meshes(meshes)


def merge_meshes(meshes):
    merged = collada.Collada()
    if len(meshes) == 0:
        return merged

    merged.assetInfo = meshes[0].assetInfo

    scene_nodes = []
    for mesh in meshes:
        merged.geometries.extend(mesh.geometries)

        for scene in mesh.scenes:
            scene_nodes.extend(scene.nodes)

    myscene = collada.scene.Scene("myscene", scene_nodes)
    merged.scenes.append(myscene)
    merged.scene = myscene

    return merged


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python sdf2dae.py <model_name> <output.dae>')
        exit()

    models_path = [
        # '/home/ruslan/.gazebo/models/',
        '/home/ruslan/.ignition/fuel/fuel.ignitionrobotics.org/openrobotics/models/'
    ]

    # parse_world(sys.argv[1])

    output = parse_model(sys.argv[1])
    output.write(sys.argv[2])
