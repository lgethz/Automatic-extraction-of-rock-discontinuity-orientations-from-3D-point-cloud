# DualProjectionNormalEstimation
## 1. Function
Normal vector estimation method for point clouds with the ability to preserve sharp features.
## 2. File Description
- main.py: Entry point of the program
- dataset.py: Responsible for dataset loading
- dppca.py: Core implementation — normal estimation based on double-projection PCA
- octreebaseddownsampling.py: Octree downsampling used before normal estimation
- skeleton_improved.py: Point cloud skeleton extraction
- utils.py: Utility functions
- requirements.txt: Environment dependencies required to run the program
## 3. Command Line Usage
```bash
python main.py \
    --dataset_root (Root directory of the dataset) \
    --data_set (Dataset directory) \
    --log_root (Directory for saving logs) \
    --evaluate (Whether to perform evaluation; only available when ground truth exists) \
    --data_list (Name of the data list) \
    --downsample true \
    --downsample_max_depth 8 \
    --downsample_voxel_size (Optional target leaf width in point-cloud coordinate units) \
    --downsample_plot false \
    --k_pca (Number of neighbors for PCA) \
    --k_dppca (Number of neighbors for DP-PCA) \
    --k_recomputation (Number of neighbors for recomputation) \
    --threshold_dummy (Threshold for identifying corner points) \
    --num_bin (Number of bins in histogram) \
    --n_jobs (Number of parallel processes) \
    --show_res (Whether to visualize results) \
    --save_normal_xyz (Whether to save normal vectors as xyz format) \
    --show_dummy (Whether to display invalid points) \
    --sigma_threshold (Sigma threshold parameter) \
    --debug (Enable debug mode)
```
When run directly, logs default to the project-level `logs` folder. The
top-level pipeline also uses that folder unless `--log-root` is specified.
Octree downsampling is opt-in to preserve the previous full-resolution
behavior. Set `--downsample true` to enable it. `--downsample_max_depth` sets
the maximum subdivision level. Optionally, `--downsample_voxel_size` selects a
target leaf width; when both are supplied, the depth is capped by the maximum
depth. The downsampled XYZ cloud is saved under the run's `downsampled`
directory and is the cloud used for normal estimation. Keep downsampling off
when using `--evaluate true`, since ground-truth normals correspond to the
original points. Set `--downsample_plot true` to compare the original cloud
(left) with the downsampled cloud (right), including each cloud's point count
in the viewer title, before normal estimation; close the viewer to continue.
## 4.Updates
More related source code will be updated in the future.

## 5. Example （video）
The sample video is located at  ```./example_video/example.mp4.```