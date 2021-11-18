## Prerequisites

Before running the demo, make sure you followed the installation instructions in
[install.md](https://github.com/tpet/rpz_planning/blob/master/docs/install.md)

## Singularity

Simply run the command:
```bash
./singularity/rpz_planning.sif
```

Another option, is to log into the container mounting your catkin workspace with the package:
```bash
singularity shell --nv --bind $HOME/trajopt_ws/src/:/opt/ros/trajopt_ws/src/ rpz_planning.sif
```

From the singularity container:
```bash
source /opt/ros/melodic/setup.bash && \
source /opt/ros/cras_subt/devel/setup.bash --extend && \
source /opt/ros/trajopt_ws/devel/setup.bash --extend && \
roslaunch rpz_planning play.launch bag:=$HOME/subt/trajopt_ws/src/rpz_planning/data/marv_2021-04-29-12-48-13.bag
```

## Locally

The trajectory optimization takes into account traversability information
(local map roll, pitch and height values), as well as observation rewards.

Download an example bag-file from
[http://ptak.felk.cvut.cz/darpa-subt/data/rpz_planning/bags/trajopt_input/marv_2021-04-29-12-48-13.bag](http://ptak.felk.cvut.cz/darpa-subt/data/rpz_planning/bags/trajopt_input/marv_2021-04-29-12-48-13.bag)
and place it in the `rpz_planning/data` directory. Then launch the demo:

```bash
roslaunch rpz_planning play.launch
```

