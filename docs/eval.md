## Exploration and Mapping evaluation

* **Obtain ground truth**
  
  Ground truth map from the simulator could be represented as a mesh file.
  Download meshes of som of the worlds from this link:
  [https://drive.google.com/drive/folders/1eB8sJmN4EknR7cjrke248aRFPaif8srg?usp=sharing](https://drive.google.com/drive/folders/1eB8sJmN4EknR7cjrke248aRFPaif8srg?usp=sharing)
  And place them in `rpz_planning/data/meshes/` folder.  

  If you would like to create ground truth mesh by your own,
  plaese, follow instructions in
  [scripts/utils/world_to_mesh/readme.md](https://github.com/tpet/rpz_planning/blob/master/scripts/utils/world_to_mesh/readme.md)
  
* **Do mapping**

  Record the localization (`tf`) and point cloud data by adding the arguments to the previous command:
  (here the recorded bag-file duration is given in seconds).
  ```bash
  roslaunch rpz_planning naex.launch follow_opt_path:=true do_recording:=true
  ```
  
* **Compare map to mesh**
  
  (Note, that this node requires `python3` as an interpreter and `Pytorch3d` installed.
  Please, follow the
  [instructions](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md)
  to install its dependencies)

  Ones the point cloud mapping is complete, reconstruct the explored global map.
  It will wait for you to hit `space` button in order to start the bag-file.
  ```bash
  roslaunch rpz_planning eval.launch world_name:=simple_cave_03 bag:=<path/to/bag/file/bag_file_name>.bag
  ```
  Ones a ground truth mesh is loaded, start the playing the bag file (press `space` button)
  (specify the `world_name` argument to match the name of downloaded or created ground truth mesh).
  
  It will compare a point cloud to a mesh using the following functions:
  - the closest distance from
    [point to mesh edge](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_edge_distance)
    (averaged across all points in point cloud),
  - the closes distance from
    [point to mesh face](https://pytorch3d.readthedocs.io/en/latest/modules/loss.html#pytorch3d.loss.point_mesh_face_distance)
    (averaged across all points in point cloud).
    
  Evaluation metrics are combined in the ROS message,
  [Metrics.msg](https://github.com/tpet/rpz_planning/blob/8e15fde74fb64cb1d1da46e93cd446e563bfa3d6/msg/Metrics.msg):
  
  ```
  Header header
  float64 exp_face_loss
  float64 exp_edge_loss
  float64 exp_chamfer_loss
  float64 exp_completeness
  float64 map_face_loss
  float64 map_edge_loss
  float64 map_chamfer_loss
  float64 artifacts_exp_completeness
  float64 dets_score
  float64 total_reward
  float64 artifacts_total_reward
  ```
  
  Exploration metrics:
  
  - ***exp_face_loss***: distance from ground truth mesh faces to corresponding closes points in the constructed cloud,
    averaged by the number of faces.
           
  - ***exp_edge_loss***: distance from ground truth mesh edges to corresponding closes points in the constructed cloud,
    averaged by the number of edges.
    
  - ***exp_chamfer_loss***: distance from points of ground truth cloud to corresponding closes points in the constructed
    cloud, averaged by the number of ground truth points.
    
  - ***exp_completeness***: fraction of ground truth points sampled from world mesh considered as covered
    over the whole numer of ground truth world points (takes values between 0 and 1).
    A ground truth point is considered as covered if
    there is a point from the constructed map that is located within a distance threshold
    from the ground truth point.
    

  Mpping accuracy metrics:
  
  - ***map_face_loss***: distance from points in the constructed cloud to corresponding closest mesh faces, averaged by
    the number of points.
    
  - ***map_edge_loss***: distance from points in the constructed cloud to corresponding closest mesh edges, averaged by
    the number of points.
    
  - ***map_chamfer_loss***: distance from points of constructed cloud to corresponding closes points in the ground truth
    cloud, averaged by the number of constructed points.
    

  Rewards based on ground truth map and artifacts coverage:
  
  - ***artifacts_exp_completeness***: fraction of ground truth points sampled from artifacts' meshes considered as covered
    over the whole numer of ground truth artifacts' points (takes values between 0 and 1).
    A ground truth point is considered as covered if
    there is a point from the constructed map that is located within a distance threshold
    from the ground truth point.
    
  - ***dets_score***: (takes values between 0 and N_artifacts) number of correctly detected artifacts from
    confirmed hypothesis.
    A detected artifact is considered being correctly detected if its most probable class corresponds to the
    ground truth class and the detection is located within a distance threshold
    from the ground truth artifact pose.
    
  - ***total_reward***: accumulated reward for the whole exploration route based on visibility information:
      - 1-5 m distance range range,
      - points occlusion,
      - points that are inside cameras FOVs.
    
  - ***artifacts_total_reward***: similar to `total_reward` metric, but computed using the artifacts' covered points.
  
  
  Main parameters for the evaluation could be specified in a launcher,
  [eval.launch](https://github.com/tpet/rpz_planning/blob/dcd3948d170746dc381e1be9a9f784de2b648cbd/launch/eval.launch#L58).

