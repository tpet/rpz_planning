#!/bin/bash

# Convert a Gazebo world to a mesh file
# The .obj file can later be processed by Pytorch3d library

in=$1
out=$2

if [ -z $3 ]; then
  axis_up="Z"
else
  axis_up=$3
fi

__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$out" ] || [ -z "$in" ]; then
  echo 'Convert a Gazebo world to an octomap file compatible with the octomap ROS node\n'
  echo 'Usage: world2oct.sh <input_file.world> <output_file.obj> [axis_up "Z"|"Y"]'
  exit
fi

if [ ! -e $HOME/opt/binvox ]; then
  echo "Unable to locate the binvox executable in the $HOME folder !"
fi

rm /tmp/file.dae /tmp/file.obj

# Convert from Gazebo world to Collada .dae
python3 ${__dir}/world2dae.py $in /tmp/file.dae $axis_up

# Convert from Collada .dae to .obj using Blender
blender --background --python ${__dir}/dae_to_obj.py -- /tmp/file.dae $out $axis_up

