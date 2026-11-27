# Discontinuity Analysis

This pipeline downsamples the point cloud using a octree-based downsampling approach, estimates point-cloud normals, then groups their orientations,
segments planes and calculates discontinuity orientations.

## Source

This code is mainly based on:
- Normal-vector estimation is based on
  [NormalVecotrEstimation_dppca](https://github.com/MingmingRen0616/NormalVecotrEstimation_dppca).
- Rock discontinuity extraction is based on
  [RockDiscontinuityExtraction](https://github.com/MingmingRen0616/RockDiscontinuityExtraction).

Ren, M., Hu, J., Peng, D., Ding, Y., & He, M. (2025). Automatic extraction of
rock discontinuity orientations from 3D point clouds via an adaptive clustering
and surface fitting approach. *International Journal of Rock Mechanics and
Mining Sciences, 194*, 106246.

## Requirements and setup (For Windows)

Use Python 3.10. In PowerShell, go to the repository root, create and activate
a virtual environment, then install the dependencies:

```powershell
cd "C:\path\to\DiscontinuityAnalysis"
py -3.10 -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass) # Only if error pops up after step before
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If using conda, use following command to to the repository root, create and activate
a virtual environment, then install the dependencies:

```powershell
cd "C:\path\to\DiscontinuityAnalysis"
conda create --name discontinuity_env python=3.10 -y
conda activate discontinuity_env
pip install -r requirements.txt
```

## Input data

Use `--data-path` to provide either one `.xyz` file or a directory containing
`.xyz` files directly inside it. A directory is processed in filename order;
subdirectories and non-`.xyz` files are ignored. No dataset folders or list
file need to be created.

Each `.xyz` file must contain numeric data with at least three columns in this
order: `Easting (X) Northing (Y) Elevation (Z)`. Extra columns are ignored by
normal estimation. Files ending in `_gt.xyz` are treated as ground-truth
sidecars, not input clouds. With `--norm-evaluate true`, each input needs a
neighboring `<name>_gt.xyz` file.

Dip direction is reported as an azimuth clockwise from North (+Y) toward East
(+X): 0° = North, 90° = East, 180° = South, and 270° = West. For horizontal
planes, dip direction is undefined and is written as `nan`.

Areas, where no points are present in the point cloud, will not be recognized
or measured by the code.


## Run

Run this command in PowerShell from the repository root. It uses the paths and
parameter values shown below:

```powershell
python .\main.py --data-path "C:\Users\...\point_clouds" --cut-cloud true --log-root "C:\Users\...\DiscontinuityAnalysis\logs" --norm-evaluate false --norm-downsample true --norm-downsample-max-depth 8 --norm-downsample-voxel-size 0.02 --norm-downsample-plot true --norm-k-pca 45 --norm-k-dppca 100 --norm-k-recomputation 40 --norm-threshold-dummy 0.7 --norm-num-bin 150 --norm-n-jobs 4 --norm-show-res true --norm-save-normal-xyz true --norm-show-dummy false --norm-sigma-threshold 0.05 --norm-debug false --rock-res-root "C:\Users\...\DiscontinuityAnalysis\results" --rock-data-suffix ".xyz" --rock-file-suffix ".txt" --rock-angle 25 --rock-iter 10 --rock-min-clusters 50 --rock-show-cluster-res false --rock-save-cluster-res true --rock-show-planes-res true --rock-show-fitted-planes true --rock-save-planes-res true --rock-save-orientation-res true
```

An easier way of running the code is directly in START.py, where the parameters can
be edited directly in e.g. VScode and hit on run directly.

The pipeline runs normal estimation once for the `.xyz` file or for every `.xyz` file
in the input directory, then extracts planes and orientations from each result.

The stereoplots shows the plane poles of the segemented planes with Fisher density contours,
clockwise from North (+Y) toward East (+X). All the data is saved in the results folder

Update the absolute paths if the project is in a different folder. Use
`python .\main.py --help` to see the command-line options.

## Parameters

All Boolean options accept `true` or `false`. Required options have no default.
Paths are relative to the current directory unless given as absolute paths.

| Option | Default | Description |
|---|---:|---|
| `--data-path` | required | Input `.xyz` file or directory containing `.xyz` files. |
| `--cut-cloud` | `false` | Interactively crop each input cloud before analysis. |
| `--log-root` | `.\logs` | Root directory for logs from both stages. |
| `--norm-evaluate` | `false` | Evaluate estimated normals against ground truth. |
| `--norm-downsample` | `false` | Octree-downsample before estimating normals. |
| `--norm-downsample-voxel-size` | `none` | Target octree leaf width in point-cloud units. Larger voxel size increases downsampling.  |
| `--norm-downsample-max-depth` | `8` | Maximum octree subdivision depth. Higher subdivision depth decreases downsampling. |
| `--norm-downsample-plot` | `false` | Display downsampling comparison before estimation. |
| `--norm-k-pca` | `50` | PCA neighborhood size. |
| `--norm-k-dppca` | `250` | DP-PCA neighborhood size. |
| `--norm-k-recomputation` | `50` | Neighborhood size for normal recomputation. |
| `--norm-threshold-dummy` | `0.7` | Threshold for identifying corner points. |
| `--norm-num-bin` | `150` | Number of bins in the feature histogram. |
| `--norm-n-jobs` | `4` | Number of parallel worker processes. Reduce if the analysis crashes. |
| `--norm-show-res` | `false` | Display normal-estimation results. |
| `--norm-save-normal-xyz` | `true` | Save normals for the extraction stage; must remain true. |
| `--norm-show-dummy` | `false` | Display invalid points during evaluation. |
| `--norm-sigma-threshold` | `0.05` | Sigma threshold for feature selection. |
| `--norm-debug` | `false` | Enable normal-estimation debug output. |
| `--rock-res-root` | `.\results` | Directory for extraction results. |
| `--rock-data-suffix` | `.xyz` | Input file suffix for extraction. |
| `--rock-file-suffix` | `.txt` | Suffix for extraction result files. |
| `--rock-angle` | `25` | Angle threshold in degrees for grouping similar normals. |
| `--rock-iter` | `5` | Number of clustering iterations. |
| `--rock-min-clusters` | `100` | Minimum points required for a plane segment. |
| `--rock-show-cluster-res` | `false` | Display clustering results. |
| `--rock-save-cluster-res` | `true` | Save clustering results. |
| `--rock-show-planes-res` | `false` | Display segmented planes. |
| `--rock-show-fitted-planes` | `false` | Display fitted plane grids over their point segments before the stereoplot. |
| `--rock-save-planes-res` | `true` | Save segmented planes. |
| `--rock-save-orientation-res` | `true` | Save extracted orientations. |

### Example

```powershell
python .\main.py --data-path "C:\Users\...\Terrace.xyz" --cut-cloud True --log-root "C:\Users\GuebeliL\...\logs" --norm-evaluate false --norm-downsample true --norm-downsample-max-depth 8 --norm-downsample-voxel-size 0.02 --norm-downsample-plot false --norm-k-pca 15 --norm-k-dppca 350 --norm-k-recomputation 20 --norm-threshold-dummy 0.7 --norm-num-bin 150 --norm-n-jobs 4 --norm-show-res true --norm-save-normal-xyz true --norm-show-dummy false --norm-sigma-threshold 0.05 --norm-debug false --rock-res-root "C:\Users\...\results" --rock-data-suffix ".xyz" --rock-file-suffix ".txt" --rock-angle 25 --rock-iter 4 --rock-min-clusters 30 --rock-show-cluster-res false --rock-save-cluster-res true --rock-show-planes-res true --rock-show-fitted-planes true --rock-save-planes-res true --rock-save-orientation-res true
```

## Output

Extraction results are written to `.\results\<data-set>\<timestamp>\` by
default. With `--data-path`, `<data-set>` is the input file's stem or the
input directory's name. The output files use comma-separated columns; the
first line is a commented column header.

| File | Contents |
|---|---|
| `all_planes.txt` | Retained points and their normals: `x, y, z, nx, ny, nz, orientation_group_id, plane_segment_id`. Written when `--rock-save-planes-res true`. |
| `parameters.txt` | Fitted plane equations: `normal_a, normal_b, normal_c, plane_offset_d, plane_id`, for `a*x + b*y + c*z + d = 0`. |
| `orientations.txt` | Fitted plane orientations: `dip_degrees, dip_direction_degrees, plane_id`. |
| `orientations_<name>_stereoplot.png` | Lower-hemisphere equal-angle stereoplot with Fisher density contours, opened in the default browser. |
| `<count>_clustering_analysis.txt` | Points and normals assigned to orientation groups: `x, y, z, nx, ny, nz, orientation_group_id`. Written when `--rock-save-cluster-res true`. |
| `segmented_planes.png` | Screenshot of the segmented-plane viewer, if `--rock-show-planes-res true`. |

`parameters.txt` and `orientations.txt` are written when
`--rock-save-orientation-res true`. The plane and cluster file suffixes can be
changed with `--rock-file-suffix`.



## Empirical Parameter Guidelines

For a point cloud with a point spacing (resolution) of **0.5–1.0 cm**, use the following baseline parameters as a starting point:

| Parameter | Baseline | Adjustment Guideline |
| :--- | :---: | :--- |
| `--norm-downsample-max-depth` | `8` | Decrease, or disable downsampling (`--norm-downsample false`), if the input resolution is already low. |
| `--norm-downsample-voxel-size` | `0.02` | Decrease, or disable downsampling (`--norm-downsample false`), if the input resolution is already low. |
| `--norm-k-pca` | `15` | Decrease if the level of detail is high, small-scale features are present, or the resolution is low relative to the discontinuity scale. |
| `--norm-k-dppca` | `350` | Decrease if the level of detail is high, small-scale features are present, or the resolution is low relative to the discontinuity scale. |
| `--norm-k-recomputation` | `20` | Decrease if the level of detail is high, small-scale features are present, or the resolution is low relative to the discontinuity scale. |
| `--norm-threshold-dummy` | `0.7` | Increase (e.g., `0.8–0.9`) to apply stricter filtering to edge points. |
| `--norm-sigma-threshold` | `0.05` | Increase (e.g., `0.08-0.10`) to makes the normal-estimation stage less sensitive to surface curvature. |
| `--norm-num-bin` | `150` | Recommended to keep at the default value of `150`. |
| `--rock-angle` | `30` | Decrease if multiple discontinuity sets have similar orientations to prevent them from merging. |
| `--rock-min-clusters` | `30` | Decrease if the level of detail is high, small-scale features are present, or the resolution is low relative to the discontinuity scale. |

> **Note:** Always fine-tune these values according to the point cloud resolution, surface roughness, and the scale of the target geological features. You can check the fit of the planes when turning on `--rock-show-fitted-planes true`. It also helps to show the segemented planes using `--rock-show-planes-res true`.