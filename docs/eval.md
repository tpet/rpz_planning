## Exploration and Mapping evaluation

* **Obtain ground truth**
  
  Ground truth map from the simulator could be represented as a mesh file.
  Download meshes of som of the worlds from this link:
  [https://drive.google.com/drive/folders/1eB8sJmN4EknR7cjrke248aRFPaif8srg?usp=sharing](https://drive.google.com/drive/folders/1eB8sJmN4EknR7cjrke248aRFPaif8srg?usp=sharing)
  And place them in `rpz_planning/data/meshes/` folder.  

  If you would like to create ground truth mesh by your own,
  plaese, follow instructions in
  [tools/world_to_mesh/readme.md](https://github.com/tpet/rpz_planning/blob/master/tools/world_to_mesh/readme.md)
  
* **Do mapping**

  Record the localization (`tf`) and point cloud data by adding the arguments to the previous command:
  (here the recorded bag-file duration is given in seconds).
  ```bash
  roslaunch rpz_planning naex.launch follow_opt_path:=true do_recording:=true duration:=120
  ```
  
* **Compare map to mesh**
  
  (Note, that this node requires `python3` as an interpreter.
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
  [map_eval.launch](https://github.com/tpet/rpz_planning/blob/8e15fde74fb64cb1d1da46e93cd446e563bfa3d6/launch/map_eval.launch#L45):

  ```xml
  <rosparam subst_value="true">
      map_topic: $(arg map_topic)  <!-- Point cloud map constructed during exploration -->
      gt_mesh: $(arg gt_mesh_file) <!-- Path to .obj or .ply ground truth mesh file of the world -->
      do_points_sampling_from_mesh: True <!-- Whether to use points sampling method from mesh (True) or use mesh vertices (False) -->
      do_points_sampling_from_map: True <!-- Whether to subsample constructed map before metrics computation -->
      n_sample_points: 10000 <!-- Max number of sampled points that are used both from gt mesh and constructed map if above params are True -->
      do_eval: $(arg do_eval) <!-- Whether to run evaluation function or just see the constructed map -->
      load_gt: $(arg load_gt) <!-- Whether to load ground truth mesh of the environment -->
      coverage_dist_th: 0.5 <!-- Threshold value [m]: defines if a world mesh point is considered covered -->
      artifacts_coverage_dist_th: 0.1 <!-- Threshold value [m]: defines if an artifact mesh point is considered covered -->
      artifacts_hypothesis_topic: "detection_localization/dbg_confirmed_hypotheses_pcl" <!-- Detections localization output to evaluate -->
      detections_dist_th: 1.5 <!-- Threshold value [m]: defines if an artifact detection point is considered close to gt pose -->
      max_age: 60.0 <!-- Constructed map point cloud msg timestamp threshold: it wouldn't be processed if dt > age -->
      map_gt_frame: $(arg world_name) <!-- Ground truth world mesh frame: usually it is the same as world name in Subt simulator -->
      eval_rate: 0.03 <!-- How frequent to perform evaluation -->
      record_metrics: $(arg do_metrics_record) <!-- Whether to record obtained metrics to .xls file -->
      metrics_xls_file: $(arg bag) <!-- Path to save metrics .xls file -->
  </rosparam>
  ```
