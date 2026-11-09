

Input:

python .\main.py `
  --dataset-root "C:\path\to\dataset" `             # Dataset root; contains the dataset and data-list folders.
  --data-set "MyDataset" `                          # Dataset name; shared by both processing stages.
  --log-root "C:\path\to\logs" `                    # Root folder for logs from both stages (defaults to .\logs).
  --norm-data-list "MyList" `                       # Required list name; identifies which point clouds to process.
  --norm-evaluate false `                           # Compare predicted normals with ground truth; requires ground-truth data.
  --norm-downsample true `                          # Octree-downsample before estimating normals.
  --norm-downsample-max-depth 8 `                   # Maximum octree subdivision depth.
  --norm-downsample-voxel-size 0.02 `               # Optional target leaf width in coordinate units.
  --norm-downsample-plot false `                    # Display each downsampled cloud before normal estimation.
  --norm-k-pca 50 `                                 # Number of neighbors used for PCA.
  --norm-k-dppca 250 `                              # Number of neighbors used for DP-PCA.
  --norm-k-recomputation 50 `                       # Number of neighbors used for normal recomputation.
  --norm-threshold-dummy 0.7 `                      # Threshold used to identify corner points.
  --norm-num-bin 150 `                              # Number of bins used in the feature histogram.
  --norm-n-jobs 4 `                                 # Number of parallel worker processes.
  --norm-show-res false `                           # Display the normal-estimation result.
  --norm-save-normal-xyz true `                     # Save predicted normals; must be true for the extraction stage.
  --norm-show-dummy false `                         # Display invalid points when evaluation is enabled.
  --norm-sigma-threshold 0.05 `                     # Sigma threshold used during feature selection.
  --norm-debug false `                              # Enable normal-estimation debug output.
  --rock-res-root "C:\path\to\results" `            # Folder where extraction results are saved.
  --rock-data-suffix ".xyz" `                       # Input file suffix; .xyz matches the normal-estimation output.
  --rock-file-suffix ".txt" `                       # Suffix for extraction result files.
  --rock-angle 38 `                                 # Angle threshold for plane similarity.
  --rock-iter 5 `                                   # Number of clustering iterations.
  --rock-min-clusters 100 `                         # Minimum number of points required for a plane cluster.
  --rock-show-cluster-res false `                   # Display clustering results.
  --rock-save-cluster-res true `                    # Save clustering results.
  --rock-show-planes-res false `                    # Display segmented planes.
  --rock-save-planes-res true `                     # Save segmented planes.
  --rock-save-orientation-res true                  # Save extracted orientations.


Octree downsampling is opt-in to preserve the previous full-resolution
behavior. Set `--norm-downsample true` to enable it. If both downsampling
controls are set, the maximum depth caps the depth selected from the target
voxel size. The downsampled XYZ points are used by both normal estimation and
the subsequent rock-extraction stage. Keep downsampling off when evaluating
against ground truth at the original point locations.
Set `--norm-downsample-plot true` to display each downsampled cloud before
normal estimation. The viewer places the original cloud on the left and the
downsampled cloud on the right; close it to continue processing.