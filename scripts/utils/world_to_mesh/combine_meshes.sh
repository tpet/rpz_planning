#! /bin/bash

input_dir=$1
out_file=$2

if [ -z "$out_file" ] || [ -z "$input_dir" ]; then
  echo 'Usage: ./combine_meshes.sh <input_path_to_daes> <output_file.dae>'
  exit
fi

__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo $input_dir
echo $out_file

blender --background --python ${__dir}/blend_import_meshes.py -- ${input_dir} ${out_file}
