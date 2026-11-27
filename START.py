"""
Discontinuity Analysis  ·  START

Set your parameters below, then press Run in VS Code or another programme.
Details on every parameter: see README.md
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
#  1 · INPUT / OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH   = r"C:\Users\...\data"          # one .xyz file or a folder of .xyz files
CUT_CLOUD   = False                         # crop the cloud interactively first

RESULTS_DIR = HERE / "results"
LOGS_DIR    = HERE / "logs"


# ══════════════════════════════════════════════════════════════════════════════
#  2 · NORMAL ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

# Downsampling
DOWNSAMPLE      = True    # octree downsampling before estimation
VOXEL_SIZE      = 0.02    # leaf width in cloud units (None = use max depth only)
MAX_DEPTH       = 8       # max octree depth (higher = less downsampling)

# Neighbourhood sizes
K_PCA           = 15
K_DPPCA         = 350
K_RECOMPUTATION = 20

# Filtering
THRESHOLD_DUMMY = 0.7     # corner-point filter (higher = stricter)
SIGMA_THRESHOLD = 0.05    # curvature sensitivity (higher = less sensitive)
NUM_BIN         = 150     # keep at 150

N_JOBS          = 4       # parallel workers (reduce if the run crashes)


# ══════════════════════════════════════════════════════════════════════════════
#  3 · DISCONTINUITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

ANGLE           = 30      # angle threshold [deg] for grouping similar normals
ITERATIONS      = 5       # clustering iterations
MIN_CLUSTER     = 30      # minimum points per plane segment


# ══════════════════════════════════════════════════════════════════════════════
#  4 · VIEWERS  (windows that open during the run)
# ══════════════════════════════════════════════════════════════════════════════

SHOW_DOWNSAMPLING   = False   # compare cloud before / after downsampling
SHOW_NORMALS        = True    # normal-estimation result
SHOW_CLUSTERS       = False   # orientation clusters
SHOW_PLANES         = True    # segmented planes
SHOW_FITTED_PLANES  = True    # fitted plane grids over their segments


# ══════════════════════════════════════════════════════════════════════════════
#  5 · ADVANCED  (normally left as is)
# ══════════════════════════════════════════════════════════════════════════════

NORM_EVALUATE       = False   # needs <name>_gt.xyz; not with downsampling / cropping
NORM_DEBUG          = False
NORM_SHOW_DUMMY     = False
SAVE_CLUSTERS       = True
SAVE_PLANES         = True
SAVE_ORIENTATIONS   = True    # must be True to get orientations + stereoplot


# ══════════════════════════════════════════════════════════════════════════════
#  Nothing to edit below this line
# ══════════════════════════════════════════════════════════════════════════════

def _flag(name, value):
    """Turn one setting into a command-line flag pair for main.py."""
    if isinstance(value, bool):
        value = str(value).lower()
    return [f"--{name}", str(value)]


def build_arguments():
    argv = [
        *_flag("data-path", DATA_PATH),
        *_flag("cut-cloud", CUT_CLOUD),
        *_flag("log-root", LOGS_DIR),
        *_flag("rock-res-root", RESULTS_DIR),

        *_flag("norm-downsample", DOWNSAMPLE),
        *_flag("norm-downsample-max-depth", MAX_DEPTH),
        *_flag("norm-downsample-plot", SHOW_DOWNSAMPLING),
        *_flag("norm-k-pca", K_PCA),
        *_flag("norm-k-dppca", K_DPPCA),
        *_flag("norm-k-recomputation", K_RECOMPUTATION),
        *_flag("norm-threshold-dummy", THRESHOLD_DUMMY),
        *_flag("norm-sigma-threshold", SIGMA_THRESHOLD),
        *_flag("norm-num-bin", NUM_BIN),
        *_flag("norm-n-jobs", N_JOBS),
        *_flag("norm-show-res", SHOW_NORMALS),
        *_flag("norm-evaluate", NORM_EVALUATE),
        *_flag("norm-debug", NORM_DEBUG),
        *_flag("norm-show-dummy", NORM_SHOW_DUMMY),
        *_flag("norm-save-normal-xyz", True),   # required by the extraction stage

        *_flag("rock-angle", ANGLE),
        *_flag("rock-iter", ITERATIONS),
        *_flag("rock-min-clusters", MIN_CLUSTER),
        *_flag("rock-show-cluster-res", SHOW_CLUSTERS),
        *_flag("rock-show-planes-res", SHOW_PLANES),
        *_flag("rock-show-fitted-planes", SHOW_FITTED_PLANES),
        *_flag("rock-save-cluster-res", SAVE_CLUSTERS),
        *_flag("rock-save-planes-res", SAVE_PLANES),
        *_flag("rock-save-orientation-res", SAVE_ORIENTATIONS),
    ]
    if VOXEL_SIZE is not None:
        argv += _flag("norm-downsample-voxel-size", VOXEL_SIZE)
    return argv


def main():
    # Keep box-drawing / degree characters readable in the Windows terminal
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if not Path(DATA_PATH).expanduser().exists():
        sys.exit(f"\nInput not found: {DATA_PATH}\nSet DATA_PATH at the top of START.py.\n")

    sys.path.insert(0, str(HERE))
    import main as pipeline

    # Reuse the pipeline's own parser, so defaults and checks stay identical
    sys.argv = [str(HERE / "main.py"), *build_arguments()]
    args = pipeline.parse_arguments()

    print("\n  Discontinuity Analysis")
    print("  " + "─" * 40)
    print(f"  Input    {DATA_PATH}")
    print(f"  Results  {RESULTS_DIR}")
    print("  " + "─" * 40 + "\n")

    pipeline.run_pipeline(args)


if __name__ == "__main__":
    main()