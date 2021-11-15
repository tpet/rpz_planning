## Prerequisites

Setup access to Gitlab workspace. Follow the instructions in the doc:

[How to - GitLab and setting up workspace](https://docs.google.com/document/d/1Jwnu1jSB3GD0ZptfKwZy1fdjjVrTzuNYB_ebzWgul9U/edit#)

## Setup

Setup ROS workspace with the dependencies:

- [`dem_predictor`](https://bitbucket.org/salanvoj/dem_predictor/src/stable/):
  ROS node for traversability estimation from point cloud input
  (make sure to clone the version from `stable` branch),
- [`nifti_vision_data`](https://gitlab.fel.cvut.cz/cras/subt/tradr-robot/tradr-ugv-base/-/tree/master/):
  ROS node for dealing with `/tf_static` problems with rosbags.

```bash
mkdir -p ~/trajopt_ws/src/ && cd ~/trajopt_ws/src/

git clone -b sim https://bitbucket.org/salanvoj/dem_predictor.git

git clone https://gitlab.fel.cvut.cz/cras/subt/tradr-robot/tradr-ugv-base/

git clone https://github.com/tpet/rpz_planning/
```

Build the packages in workspace:

```bash
cd ~/trajopt_ws
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
