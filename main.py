import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
NORMAL_MAIN = PROJECT_ROOT / "1NormalVectorEstimation" / "main.py"
ROCK_MAIN = PROJECT_ROOT / "2RockDiscontinuityExtraction" / "main.py"


def parse_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected true or false."
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run normal estimation, followed by rock discontinuity extraction."
    )

    # Shared options
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "dataset"),
        help="Root directory containing the input dataset.",
    )
    parser.add_argument(
        "--data-set",
        required=True,
        help="Dataset directory/name used by both stages.",
    )
    parser.add_argument(
        "--log-root",
        default=str(PROJECT_ROOT / "logs"),
        help="Root directory for logs from both stages.",
    )

    # Normal-estimation options
    parser.add_argument("--norm-data-list", required=True, help="Name of the normal-estimation data list.")
    parser.add_argument("--norm-evaluate", type=parse_bool, default=False)
    parser.add_argument(
        "--norm-downsample",
        type=parse_bool,
        default=False,
        help="Apply octree downsampling before normal estimation.",
    )
    parser.add_argument(
        "--norm-downsample-voxel-size",
        type=float,
        default=None,
        help="Target octree leaf width in point-cloud coordinate units.",
    )
    parser.add_argument(
        "--norm-downsample-max-depth",
        type=int,
        default=8,
        help="Maximum octree subdivision depth (default: 8).",
    )
    parser.add_argument(
        "--norm-downsample-plot",
        type=parse_bool,
        default=False,
        help="Display each downsampled cloud before normal estimation.",
    )
    parser.add_argument("--norm-k-pca", type=int, default=50)
    parser.add_argument("--norm-k-dppca", type=int, default=250)
    parser.add_argument("--norm-k-recomputation", type=int, default=50)
    parser.add_argument("--norm-threshold-dummy", type=float, default=0.7)
    parser.add_argument("--norm-num-bin", type=int, default=150)
    parser.add_argument("--norm-n-jobs", type=int, default=4)
    parser.add_argument("--norm-show-res", type=parse_bool, default=False)
    parser.add_argument("--norm-save-normal-xyz", type=parse_bool, default=True)
    parser.add_argument("--norm-show-dummy", type=parse_bool, default=False)
    parser.add_argument("--norm-sigma-threshold", type=float, default=0.05)
    parser.add_argument("--norm-debug", type=parse_bool, default=False)

    # Rock-extraction options
    parser.add_argument(
        "--rock-res-root",
        default=str(PROJECT_ROOT / "results"),
        help="Directory for extraction results.",
    )
    parser.add_argument("--rock-data-suffix", default=".xyz")
    parser.add_argument("--rock-file-suffix", default=".txt")
    parser.add_argument("--rock-angle", type=float, default=38)
    parser.add_argument("--rock-iter", type=int, default=5)
    parser.add_argument("--rock-min-clusters", type=int, default=100)
    parser.add_argument("--rock-show-cluster-res", type=parse_bool, default=False)
    parser.add_argument("--rock-save-cluster-res", type=parse_bool, default=True)
    parser.add_argument("--rock-show-planes-res", type=parse_bool, default=False)
    parser.add_argument("--rock-save-planes-res", type=parse_bool, default=True)
    parser.add_argument("--rock-save-orientation-res", type=parse_bool, default=True)

    args = parser.parse_args()

    if not args.norm_save_normal_xyz:
        parser.error(
            "--norm-save-normal-xyz must be true: the extraction stage "
            "needs the normal-estimation output file."
        )
    if args.norm_downsample and args.norm_evaluate:
        parser.error(
            "--norm-evaluate cannot be used with --norm-downsample true because "
            "ground-truth normals are at the original point locations. "
            "Use --norm-downsample false to evaluate at full resolution."
        )
    if args.norm_downsample_max_depth < 0:
        parser.error("--norm-downsample-max-depth must be non-negative.")
    if (
        args.norm_downsample_voxel_size is not None
        and args.norm_downsample_voxel_size <= 0
    ):
        parser.error("--norm-downsample-voxel-size must be positive.")
    if args.norm_downsample_plot and not args.norm_downsample:
        parser.error("--norm-downsample-plot requires --norm-downsample true.")

    return args


def run_pipeline(args):
    dataset_root = Path(args.dataset_root).resolve()
    log_root = Path(args.log_root).resolve()
    res_root = Path(args.rock_res_root).resolve()

    data_list_file = (
        dataset_root
        / args.data_set
        / args.norm_data_list
        / "list"
        / f"{args.norm_data_list}.txt"
    )
    if not data_list_file.is_file():
        raise FileNotFoundError(f"Data-list file not found: {data_list_file}")

    shape_names = [
        line.strip()
        for line in data_list_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not shape_names:
        raise ValueError(f"No point-cloud names found in: {data_list_file}")

    if not NORMAL_MAIN.is_file():
        raise FileNotFoundError(f"Normal-estimation script not found: {NORMAL_MAIN}")
    if not ROCK_MAIN.is_file():
        raise FileNotFoundError(f"Rock-extraction script not found: {ROCK_MAIN}")

    # The normal-estimation script creates its output directory with
    # os.makedirs() and expects this parent directory to exist.
    normal_log_parent = log_root / args.data_set / args.norm_data_list
    normal_log_parent.mkdir(parents=True, exist_ok=True)

    previous_runs = {
        path
        for path in normal_log_parent.iterdir()
        if path.is_dir()
    }

    normal_command = [
        sys.executable,
        str(NORMAL_MAIN),
        "--dataset_root", str(dataset_root),
        "--data_set", args.data_set,
        "--log_root", str(log_root),
        "--evaluate", str(args.norm_evaluate),
        "--data_list", args.norm_data_list,
        "--downsample", str(args.norm_downsample),
        "--downsample_max_depth", str(args.norm_downsample_max_depth),
        "--downsample_plot", str(args.norm_downsample_plot),
        "--k_pca", str(args.norm_k_pca),
        "--k_dppca", str(args.norm_k_dppca),
        "--k_recomputation", str(args.norm_k_recomputation),
        "--threshold_dummy", str(args.norm_threshold_dummy),
        "--num_bin", str(args.norm_num_bin),
        "--n_jobs", str(args.norm_n_jobs),
        "--show_res", str(args.norm_show_res),
        "--save_normal_xyz", str(args.norm_save_normal_xyz),
        "--show_dummy", str(args.norm_show_dummy),
        "--sigma_threshold", str(args.norm_sigma_threshold),
        "--debug", str(args.norm_debug),
    ]
    if args.norm_downsample_voxel_size is not None:
        normal_command.extend(
            [
                "--downsample_voxel_size",
                str(args.norm_downsample_voxel_size),
            ]
        )

    print("Running normal-vector estimation...")
    subprocess.run(
        normal_command,
        cwd=NORMAL_MAIN.parent,
        check=True,
    )

    new_runs = [
        path
        for path in normal_log_parent.iterdir()
        if path.is_dir() and path not in previous_runs
    ]
    if len(new_runs) != 1:
        raise RuntimeError(
            "Could not identify the normal-estimation output directory. "
            f"Expected one new directory, found {len(new_runs)}."
        )

    prediction_dir = new_runs[0] / "pred_normal"

    # The extraction program reads one file named
    # <data_set><data_suffix>, so run it once for each prediction.
    with tempfile.TemporaryDirectory(prefix="rock-extraction-") as temp_dir:
        staged_root = Path(temp_dir)

        for shape_name in shape_names:
            prediction_file = prediction_dir / f"{shape_name}.xyz"
            if not prediction_file.is_file():
                raise FileNotFoundError(
                    f"Expected prediction was not created: {prediction_file}"
                )

            staged_dataset_dir = staged_root / args.data_set
            staged_dataset_dir.mkdir(parents=True, exist_ok=True)

            staged_input = (
                staged_dataset_dir
                / f"{args.data_set}{args.rock_data_suffix}"
            )
            shutil.copy2(prediction_file, staged_input)

            rock_command = [
                sys.executable,
                str(ROCK_MAIN),
                "--dataset_root", str(staged_root),
                "--data_set", args.data_set,
                "--res_root", str(res_root),
                "--log_root", str(log_root),
                "--data_suffix", args.rock_data_suffix,
                "--file_suffix", args.rock_file_suffix,
                "--angle", str(args.rock_angle),
                "--iter", str(args.rock_iter),
                "--min_clusters", str(args.rock_min_clusters),
                "--show_cluster_res", str(args.rock_show_cluster_res),
                "--save_cluster_res", str(args.rock_save_cluster_res),
                "--show_planes_res", str(args.rock_show_planes_res),
                "--save_planes_res", str(args.rock_save_planes_res),
                "--save_orientation_res", str(args.rock_save_orientation_res),
            ]

            print(f"Running rock extraction for {shape_name}...")
            subprocess.run(
                rock_command,
                cwd=ROCK_MAIN.parent,
                check=True,
            )

    print("Both stages completed successfully.")


if __name__ == "__main__":
    run_pipeline(parse_arguments())