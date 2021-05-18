#!/bin/bash

# Convert a Gazebo world to an octomap file
# The .bt file can later be imported in ROS for the Octomap node

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
  echo 'Usage: world2oct.sh <input_file.world> <output_file.bt> [axis_up "Z"|"Y"]'
  exit
fi

if [ ! -e $HOME/opt/binvox ]; then
  echo "Unable to locate the binvox executable in the $HOME folder !"
fi

rm /tmp/file.dae /tmp/file.obj /tmp/file.binvox /tmp/file.binvox.bt

# Convert from Gazebo world to Collada .dae
python3 ${__dir}/world2dae.py $in /tmp/file.dae $axis_up

# Convert from Collada .dae to .obj using Blender
blender --background --python ${__dir}/blend_convert_to_obj.py -- /tmp/file.dae $out $axis_up

## Convert from .obj to voxels with binvox
## Assuming the binvox executable is in $HOME
#$HOME/opt/binvox -e -fit /tmp/file.obj
#
## Convert from .binvox to octomap .binvox.bt
#binvox2bt --mark-free /tmp/file.binvox
#
## Move the file to the expected output
#mv /tmp/file.binvox.bt $out
