#!/bin/bash

world_file=$1

if [ -z "$world_file" ]; then
  echo 'Usage: ./create_meshes.sh <world_file>'
  exit
fi
 
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rm ${__dir}/meshes/*

# Declare a ModelNames array
if [[ "$world_file" == *"finals_practice_03"* ]]; then
	declare -a ModelNames=('Cave 2 Way 01 Type A' 'Cave 3 Way 01 Type A' 'Cave 3 Way Elevation 01 Type A' 'Cave 3 Way Elevation 02 Lights Type A' 'Cave 3 Way Elevation 03 Type A' 'Cave 4 Way 01 Lights Type A' 'Cave Cap Type A' 'Cave Corner 01 Lights Type A' 'Cave Corner 02 Lights Type A' 'Cave Corner 02 Type A' 'Cave Corner 03 Type A' 'Cave Corner 04 Lights Type A' 'Cave Elevation 02 Type A' 'Cave Elevation Corner Type A' 'Cave Elevation Straight Lights Type A' 'Cave Elevation Straight Type A' 'Cave Split Type A' 'Cave Straight Lights Type A' 'Cave Straight Shift Type A' 'Cave Straight Type A' 'Cave Transition Type A to and from Type B Lights' 'Cave U Turn 01 Type A' 'Cave U Turn Elevation Lights Type A' 'Cave Vertical Shaft Cantilevered Type A' 'Cave Vertical Shaft Straight Bottom Lights Type A' 'Cave Vertical Shaft Straight Top Type A' 'Climbing Helmet With Light' 'Climbing Rope' 'Fiducial' 'Finals Staging Area' 'Fog Emitter2' 'JanSport Backpack Red' 'Large Rock Fall' 'Medium Rock Fall' 'Rescue Randy Sitting' 'Samsung J8 Black' 'SubT Challenge Cube' 'Urban Cave Transition Straight')
elif [[ "$world_file" == *"simple_cave_01"* ]]; then
	declare -a ModelNames=('Base Station' 'Cave 3 Way 01 Type B' 'Cave Cap Type B' 'Cave Cavern Split 01 Type B' 'Cave Cavern Split 02 Type B' 'Cave Corner 01 Type B' 'Cave Corner 02 Type B' 'Cave Elevation Type B' 'Cave Starting Area Type B' 'Cave Straight 01 Type B' 'Cave Straight 02 Type B' 'Cave Straight 03 Type B' 'Cave Straight 04 Type B' 'Cave Straight 05 Type B' 'Cave Vertical Shaft Type B' 'Fiducial' 'JanSport Backpack Red' 'Rescue Randy Sitting' 'Samsung J8 Black')
elif [[ "$world_file" == *"simple_cave_02"* ]]; then
	declare -a ModelNames=('Base Station' 'Cave 3 Way 01 Type B' 'Cave Cap Type B' 'Cave Corner 01 Type B' 'Cave Corner 02 Type B' 'Cave Elevation Type B' 'Cave Starting Area Type B' 'Cave Straight 01 Type B' 'Cave Straight 02 Type B' 'Cave Straight 03 Type B' 'Cave Straight 04 Type B' 'Cave Straight 05 Type B' 'Fiducial' 'JanSport Backpack Red' 'Rescue Randy Sitting' 'Samsung J8 Black')
elif [[ "$world_file" == *"simple_cave_03"* ]]; then
	declare -a ModelNames=('Base Station' 'Cave 3 Way 01 Type B' 'Cave Cap Type B' 'Cave Corner 01 Type B' 'Cave Corner 02 Type B' 'Cave Corner 30 Type B' 'Cave Corner 30F Type B' 'Cave Elevation Type B' 'Cave Starting Area Type B' 'Cave Straight 01 Type B' 'Cave Straight 02 Type B' 'Cave Straight 03 Type B' 'Cave Straight 05 Type B' 'Cave Vertical Shaft Type B' 'Fiducial' 'JanSport Backpack Red' 'Rescue Randy Sitting' 'Samsung J8 Black')
elif [[ "$world_file" == *"simple_tunnel_01"* ]]; then
	declare -a ModelNames=('Base Station' 'Fiducial' 'Jersey Barrier' 'Tunnel Tile 5' 'Tunnel Tile Blocker' 'subt_tunnel_staging_area')
elif [[ "$world_file" == *"simple_tunnel_02"* ]]; then
	declare -a ModelNames=('Base Station' 'Black and Decker Cordless Drill' 'Fiducial' 'Fire Extinguisher' 'Jersey Barrier' 'Rescue Randy Sitting' 'Samsung J8 Black' 'Tunnel Tile 1' 'Tunnel Tile 2' 'Tunnel Tile 5' 'Tunnel Tile Blocker' 'subt_tunnel_staging_area')
elif [[ "$world_file" == *"simple_tunnel_03"* ]]; then
	declare -a ModelNames=('Base Station' 'Black and Decker Cordless Drill' 'Fiducial' 'Fire Extinguisher' 'Jersey Barrier' 'Rescue Randy Sitting' 'Samsung J8 Black' 'Tunnel Tile 1' 'Tunnel Tile 2' 'Tunnel Tile 5' 'Tunnel Tile 6' 'Tunnel Tile 7' 'Tunnel Tile Blocker' 'subt_tunnel_staging_area')
elif [[ "$world_file" == *"cave_circuit_01"* ]]; then
	declare -a ModelNames=('Base Station' 'Cave 3 Way 01 Lights Type B' 'Cave 3 Way 01 Type B' 'Cave Cap Type B' 'Cave Corner 01 Lights Type B' 'Cave Corner 01 Type B' 'Cave Corner 02 Lights Type B' 'Cave Corner 02 Type B' 'Cave Elevation Lights Type B' 'Cave Elevation Type B' 'Cave Starting Area Type B' 'Cave Straight 01 Lights Type B' 'Cave Straight 01 Type B' 'Cave Straight 02 Lights Type B' 'Cave Straight 03 Type B' 'Cave Straight 04 Lights Type B' 'Cave Straight 04 Type B' 'Cave Straight 05 Lights Type B' 'Cave Straight 05 Type B' 'Cave Vertical Shaft Lights Type B' 'Climbing Helmet With Light' 'Climbing Rope' 'Fiducial' 'JanSport Backpack Red' 'Large Rock Fall' 'Medium Rock Fall' 'Rescue Randy Sitting' 'Samsung J8 Black')
elif [[ "$world_file" == *"cave_circuit_02"* ]]; then
	declare -a ModelNames=('Base Station' 'Cave 3 Way 01 Lights Type B' 'Cave 3 Way 01 Type B' 'Cave Cap Type B' 'Cave Corner 01 Lights Type B' 'Cave Corner 01 Type B' 'Cave Corner 02 Lights Type B' 'Cave Corner 02 Type B' 'Cave Elevation Lights Type B' 'Cave Elevation Type B' 'Cave Starting Area Type B' 'Cave Straight 01 Lights Type B' 'Cave Straight 01 Type B' 'Cave Straight 02 Lights Type B' 'Cave Straight 02 Type B' 'Cave Straight 03 Type B' 'Cave Straight 04 Type B' 'Cave Straight 05 Lights Type B' 'Cave Straight 05 Type B' 'Climbing Helmet With Light' 'Climbing Rope' 'Fiducial' 'JanSport Backpack Red' 'Large Rock Fall' 'Medium Rock Fall' 'Rescue Randy Sitting' 'Samsung J8 Black')
else
	echo 'Supported world names are: finals_practice_03 simple_cave_01 simple_cave_02 simple_cave_03 simple_tunnel_01 simple_tunnel_02 simple_tunnel_03'
	exit
fi

# Read the array values with space
for val in "${ModelNames[@]}"; do
	echo 'Merging meshes for:' "$val";
	python world2dae.py "$world_file" "$val"
done
