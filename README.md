# RPZ Planning

## Installation

Setup `catkin_ws` with the dependencies:

- [`dem_predictor`](https://bitbucket.org/salanvoj/dem_predictor/src/stable/):
  ROS node for traversability estimation from point cloud input
  (make sure to clone the version from `stable` branch),
- [`nifti_vision_data`](https://gitlab.fel.cvut.cz/cras/subt/tradr-robot/tradr-ugv-base/-/tree/master/):
  ROS node for dealing with `/tf_static` problems with rosbags.

```bash
mkdir -p ~/catkin_ws/src/ && cd ~/catkin_ws/src/

git clone -b stable https://bitbucket.org/salanvoj/dem_predictor.git

git clone https://gitlab.fel.cvut.cz/cras/subt/tradr-robot/tradr-ugv-base/

git clone -b eval https://github.com/tpet/rpz_planning/
```

Build the packages in workspace:

```bash
cd ~/catkin_ws
catkin build dem_predictor nifti_vision_data rpz_planning
```

## Trajectory Optimization

The trajectory optimization takes into account traversability information
(local map roll, pitch and height values), as well as observation rewards.

Download an example bag-file
from [http://ptak.felk.cvut.cz/darpa-subt/data/rpz_planning/marv_2021-04-29-12-48-13.bag](http://ptak.felk.cvut.cz/darpa-subt/data/rpz_planning/marv_2021-04-29-12-48-13.bag)
and place it in the `rpz_planning/data` directory. Then launch the demo:

```bash
roslaunch rpz_planning play.launch
```

### Exploration with trajectory optimization

Follow the instructions [here](https://docs.google.com/document/d/1Jwnu1jSB3GD0ZptfKwZy1fdjjVrTzuNYB_ebzWgul9U/edit#heading=h.kliygify8hbn)
in order to setup the Subt simulator.

Start the simulator with ROS-bridge, for example:

```bash
SUBT_WS_PATH=~/subt_ws SUBT_ROBOT_TEAM=x1 SUBT_USE_SINGULARITY=0 SUBT_HEADLESS=1 \
~/cras_subt/src/subt_virtual/scripts/run_sim worldName:=cave_circuit_01

SUBT_ROBOT_TEAM=x1 SUBT_USE_SINGULARITY=0 ~/cras_subt/src/subt_virtual/scripts/run_bridge_all
```

Run the following launch file in order to include the trajectory optimization
in the exploration pipeline:

```bash
roslaunch rpz_planning naex_opt.launch follow_opt_path:=true
```

### Mapping evaluation (in progress)

Record the localization (`tf`) and point cloud data by adding the arguments to the previous command:
(here the recorded bag-file duration is given in seconds).

```bash
roslaunch rpz_planning naex_opt.launch follow_opt_path:=true do_recording:=true duration:=30
```

Ones the recording is complete, reconstruct the explored global map:

```bash
roslaunch rpz_planning map_accumulator.launch bag:=<path to bag file>/<bag file name>.bag
```