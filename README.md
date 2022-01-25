# RPZ Planning

[![RPZ planning](./docs/demo.png)](https://youtu.be/0KzWxQjTqWM)

## Demo

This is a quick-start section, which describes how to use the `rpz_planner` as a stand-alone package
without installation of the entire exploration pipeline.
If you would like to explore, how the trajectory optimization works using a prerecorded data
(as a ROS bag-file), please, follow the instructions in
[docs/demo.md](https://github.com/tpet/rpz_planning/blob/master/docs/demo.md).

## Prerequisites

- [Singularity](https://github.com/tpet/rpz_planning/blob/master/docs/singularity.md)
- [DARPA Subt simulator](https://github.com/tpet/rpz_planning/blob/master/docs/darpa_subt.md)
- [Exloration pipeline](https://github.com/tpet/naex/blob/master/README.md)

## Exploration with trajectory optimization

Start the simulator using the simgularity image `cloudsim.simg`:

```bash
cd ~/subt_virtual/scripts/
SUBT_USE_SINGULARITY=1 SUBT_ROBOT_TEAM=x1x2 ./run_sim worldName:=simple_cave_01
```

Run the ROS-bridge to the simulator:

```bash
cd ~/subt_virtual/scripts/
SUBT_USE_SINGULARITY=1 SUBT_ROBOT_TEAM=x1x2 ./run_bridge_all worldName:=simple_cave_01
```

Run exploration pipeline
([naex](https://github.com/tpet/naex))
with local trajectory optimization:

```bash
roslaunch rpz_planning naex.launch follow_opt_path:=true
```
