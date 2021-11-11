## DARPA Subt

### Setup access to Gitlab workspace

Follow the instructions in the doc:
[How to - GitLab and setting up workspace](https://docs.google.com/document/d/1Jwnu1jSB3GD0ZptfKwZy1fdjjVrTzuNYB_ebzWgul9U/edit#)

### Subt simulator in Singularity container

Clone `subt_virtual`:
```bash
cd ~/
git clone git@gitlab.fel.cvut.cz:cras/subt/common/subt_virtual.git
```

Download [cloudsim.simg](https://gitlab.fel.cvut.cz/cras/subt/common/subt_virtual#locally-via-singularity),
which is the singularity image, that includes
[Subt simulator](https://github.com/osrf/subt) installed.

Run the simulator:
```bash
cd ~/subt_virtual/scripts/
SUBT_USE_SINGULARITY=1 ./run_sim worldName:=simple_cave_01
```
