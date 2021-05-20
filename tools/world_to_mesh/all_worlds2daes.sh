#! /bin/bash

for f in ../../data/worlds/*;
	do python world2dae.py $f ../../data/meshes/${f:7:-6}.dae; done;
