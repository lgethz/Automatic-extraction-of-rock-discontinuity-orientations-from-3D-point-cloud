# Discontinuity Analysis

This pipeline downsamples the point cloud using a octree-based downsampling approach, estimates point-cloud normals, then groups their orientations,
segments planes and calculates discontinuity orientations.

## Source

- Normal-vector estimation is based on
  [NormalVecotrEstimation_dppca](https://github.com/MingmingRen0616/NormalVecotrEstimation_dppca).
- Rock discontinuity extraction is based on
  [RockDiscontinuityExtraction](https://github.com/MingmingRen0616/RockDiscontinuityExtraction).

Ren, M., Hu, J., Peng, D., Ding, Y., & He, M. (2025). Automatic extraction of
rock discontinuity orientations from 3D point clouds via an adaptive clustering
and surface fitting approach. *International Journal of Rock Mechanics and
Mining Sciences, 194*, 106246.

## Requirements and setup

Use Python 3.10. In PowerShell, go to the repository root, create and activate
a virtual environment, then install the dependencies:

```powershell
cd "C:\path\to\DiscontinuityAnalysis"
py -3.10 -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Input data

Each input point cloud must be a numeric text `.xyz` file with at least three
columns (`x y z`). For example, with dataset root `.\dataset`, data set
`geology`, and data list `Ouray`, use this layout:

```text
dataset/
└── geology/
    └── Ouray/
        ├── ouray.xyz
        └── list/
            └── Ouray.txt
```

`Ouray.txt` contains one point-cloud name per line, without the `.xyz`
extension (for example, `ouray`). Each listed name must match an `.xyz` file
in the `Ouray` folder.


## Run

Run this command in PowerShell from the repository root. It uses the paths and
parameter values shown below:

```powershell
python .\main.py --dataset-root "C:\Users\...\DiscontinuityAnalysis\dataset" --data-set "geology" --log-root "C:\Users\...\DiscontinuityAnalysis\logs" --norm-data-list "addname" --norm-evaluate false --norm-downsample true --norm-downsample-max-depth 8 --norm-downsample-voxel-size 0.02 --norm-downsample-plot true --norm-k-pca 45 --norm-k-dppca 100 --norm-k-recomputation 40 --norm-threshold-dummy 0.7 --norm-num-bin 150 --norm-n-jobs 4 --norm-show-res true --norm-save-normal-xyz true --norm-show-dummy false --norm-sigma-threshold 0.05 --norm-debug false --rock-res-root "C:\Users\...\DiscontinuityAnalysis\results" --rock-data-suffix ".xyz" --rock-file-suffix ".txt" --rock-angle 25 --rock-iter 10 --rock-min-clusters 50 --rock-show-cluster-res false --rock-save-cluster-res true --rock-show-planes-res true --rock-save-planes-res true --rock-save-orientation-res true
```

The pipeline runs normal estimation once for every name in the list, then
extracts planes and orientations from each result. Update the absolute paths
if the project is in a different folder. Use `python .\main.py --help` to see
the command-line options.

## Parameters

All Boolean options accept `true` or `false`. Required options have no default.
Paths are relative to the current directory unless given as absolute paths.

| Option | Default | Description |
|---|---:|---|
| `--dataset-root` | `.\dataset` | Root directory of the input data. |
| `--data-set` | required | Dataset directory name. |
| `--log-root` | `.\logs` | Root directory for logs from both stages. |
| `--norm-data-list` | required | Data-list directory and list-file name. |
| `--norm-evaluate` | `false` | Evaluate estimated normals against ground truth. |
| `--norm-downsample` | `false` | Octree-downsample before estimating normals. |
| `--norm-downsample-voxel-size` | `none` | Target octree leaf width in point-cloud units. |
| `--norm-downsample-max-depth` | `8` | Maximum octree subdivision depth. |
| `--norm-downsample-plot` | `false` | Display downsampling comparison before estimation. |
| `--norm-k-pca` | `50` | PCA neighborhood size. |
| `--norm-k-dppca` | `250` | DP-PCA neighborhood size. |
| `--norm-k-recomputation` | `50` | Neighborhood size for normal recomputation. |
| `--norm-threshold-dummy` | `0.7` | Threshold for identifying corner points. |
| `--norm-num-bin` | `150` | Number of bins in the feature histogram. |
| `--norm-n-jobs` | `4` | Number of parallel worker processes. |
| `--norm-show-res` | `false` | Display normal-estimation results. |
| `--norm-save-normal-xyz` | `true` | Save normals for the extraction stage; must remain true. |
| `--norm-show-dummy` | `false` | Display invalid points during evaluation. |
| `--norm-sigma-threshold` | `0.05` | Sigma threshold for feature selection. |
| `--norm-debug` | `false` | Enable normal-estimation debug output. |
| `--rock-res-root` | `.\results` | Directory for extraction results. |
| `--rock-data-suffix` | `.xyz` | Input file suffix for extraction. |
| `--rock-file-suffix` | `.txt` | Suffix for extraction result files. |
| `--rock-angle` | `38` | Angle threshold in degrees for grouping similar normals. |
| `--rock-iter` | `5` | Number of clustering iterations. |
| `--rock-min-clusters` | `100` | Minimum points required for a plane segment. |
| `--rock-show-cluster-res` | `false` | Display clustering results. |
| `--rock-save-cluster-res` | `true` | Save clustering results. |
| `--rock-show-planes-res` | `false` | Display segmented planes. |
| `--rock-save-planes-res` | `true` | Save segmented planes. |
| `--rock-save-orientation-res` | `true` | Save extracted orientations. |

