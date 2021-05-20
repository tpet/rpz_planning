#! /bin/bash

for f in worlds/*;
	do python world2dae.py $f ./meshes/${f:7:-6}.dae; done;
