# RockDiscontinuityExtraction
## 1. Function
Automatically extract rock mass orientation information from point cloud data.
## 2. File Description
- main.py: Entry point of the program

- ClusteringAnalysis.py: Orientation group clustering analysis

- DBSCAN.py: Rock plane segmentation

- FitPlane.py: Plane fitting and orientation extraction for structural surfaces

- utils.py: Utility functions
## 3. Command Line Usage
```bash
python main.py \
    --dataset_root (Root directory of the dataset) \
    --data_set (Dataset name) \
    --res_root (Directory to save results) \
    --log_root (Directory to save logs) \
    --data_suffix (Suffix of the input data files) \
    --file_suffix (Suffix for the output files) \
    --angle (Threshold angle for plane similarity) \
    --iter (Number of clustering iterations) \
    --min_clusters (Minimum number of points per plane) \
    --show_cluster_res {True, False} \
    --save_cluster_res {True, False} \
    --show_planes_res {True, False} \
    --save_planes_res {True, False} \
    --save_orientation_res {True, False}
```
When run directly, logs default to the project-level `logs` folder. The
top-level pipeline also uses that folder unless `--log-root` is specified.
When results are saved, each result text file starts with a commented column
header. `all_planes.txt` contains one row per retained point:
`x, y, z, nx, ny, nz, orientation_group_id, plane_segment_id`. The final
column identifies which DBSCAN plane segment each point belongs to. The
segmented-plane viewer displays the retained points, using the same color for
all points in the same orientation group. When `--show_planes_res True` is
enabled, closing the viewer saves its final view as `segmented_planes.png` in
that run's results folder.
`parameters.txt` contains the fitted plane coefficients
`a, b, c, d, plane_id` for the plane equation `a*x + b*y + c*z + d = 0`.
`orientations.txt` contains `dip_degrees, dip_direction_degrees, plane_id`;
its final column maps each orientation back to the corresponding rows in
`parameters.txt` and `all_planes.txt`. Different plane IDs can have similar
orientations when separate spatial segments have similar fitted directions.
`orientation_groups.txt` contains one row per orientation group:
`group_dip_degrees, group_dip_direction_degrees, orientation_group_id, point_count`.
The dip and dip direction summarize the assigned point normals in each group;
they are not averages of the RANSAC-fitted planes in `orientations.txt`. The
group IDs correspond to the orientation-group ID column in `all_planes.txt`.
It is written when `--save_orientation_res True` is enabled. Dip direction
is an azimuth clockwise from +Y toward +X; for horizontal orientations, where
dip direction is undefined, the output is `nan`.
## 4.Updates
More related source code will be updated in the future.

## 5. Example （video）
The sample video is located at  ```./video/test.mp4.```

## Please cite our paper when using this code.
Ren, M., Hu, J., Peng, D., Ding, Y., & He, M. (2025). Automatic extraction of rock discontinuity orientations from 3D point clouds via an adaptive clustering and surface fitting approach. International Journal of Rock Mechanics and Mining Sciences, 194, 106246.