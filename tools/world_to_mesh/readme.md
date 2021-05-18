Generate a Mesh from a Gazebo world
=======================================

[Reference](https://stackoverflow.com/questions/62903784/how-to-convert-a-gazebo-world-to-a-point-cloud)

Scripts to convert a Gazebo `.world` to Mesh (`.dae` and `.obj` files).

## Dependencies

Download [Blender](https://www.blender.org/) to convert `.dae` to `.obj`.
```shell script
sudo apt install blender
```

## How to use

To convert a Gazebo world to a Mesh usable by mapping evaluation node,
use the `world2obj.sh` script. For example:
```shell script
./world2obj.sh ./worlds/warehouse.world ./meshes/warehouse.obj
```

## Workflow

Combine all DAE files with `world2dae.py` python script

DAE -> OBJ with Blender
