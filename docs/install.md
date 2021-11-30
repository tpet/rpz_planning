## [Singularity](https://github.com/tpet/rpz_planning/blob/master/docs/singularity.md)

Building Singularity image

If you would like to build a singularity image yourself,
please do the following:

```bash
cd ./singularity
sudo singularity build rpz_planning.sif rpz_planning.txt
```

## Local Setup

### Prerequisites

Setup access to Gitlab workspace. Follow the instructions in the doc:
[How to - GitLab and setting up workspace](https://docs.google.com/document/d/1Jwnu1jSB3GD0ZptfKwZy1fdjjVrTzuNYB_ebzWgul9U/edit#)

Setup ROS workspace with the dependencies:

- [`dem_predictor`](https://bitbucket.org/salanvoj/dem_predictor/src/stable/):
  ROS node for traversability estimation from point cloud input
  (make sure to clone the version from `stable` branch),
- [`nifti_vision_data`](https://gitlab.fel.cvut.cz/cras/subt/tradr-robot/tradr-ugv-base/-/tree/master/):
  ROS node for dealing with `/tf_static` problems with rosbags.

```bash
mkdir -p ~/trajopt_ws/src/ && cd ~/trajopt_ws/src/

git clone -b sim https://bitbucket.org/salanvoj/dem_predictor.git
git clone https://github.com/tpet/rpz_planning/
```

Build the packages in workspace:

```bash
cd ~/trajopt_ws
catkin build dem_predictor rpz_planning
```
