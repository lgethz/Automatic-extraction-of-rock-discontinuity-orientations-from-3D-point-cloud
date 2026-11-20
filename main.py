import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
NORMAL_MAIN = PROJECT_ROOT / "1NormalVectorEstimation" / "main.py"
ROCK_MAIN = PROJECT_ROOT / "2RockDiscontinuityExtraction" / "main.py"
STEREOPLOT_MAIN = PROJECT_ROOT / "3PlotStereoplot" / "main.py"


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


def resolve_data_path(data_path):
    path = Path(data_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".xyz":
            raise ValueError(f"Input file must have a .xyz extension: {path}")
        return path, path.stem, [path.stem]

    if not path.is_dir():
        raise FileNotFoundError(
            f"Input path does not exist or is not a file/directory: {path}"
        )

    xyz_files = sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() == ".xyz"
            and not item.stem.lower().endswith("_gt")
        ),
        key=lambda item: item.name.casefold(),
    )
    if not xyz_files:
        raise ValueError(f"No .xyz files found directly in: {path}")

    shape_names = [item.stem for item in xyz_files]
    if len({name.casefold() for name in shape_names}) != len(shape_names):
        raise ValueError(f"Input directory has duplicate .xyz file names: {path}")
    return path, path.name, shape_names


def _print_and_log(message, log_file):
    print(message, flush=True)
    log_file.write(message + "\n")
    log_file.flush()


def _run_and_log(command, cwd, log_file):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        process.kill()
        process.wait()
        raise RuntimeError("Subprocess output pipe was not created.")

    with process.stdout:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _points_in_polygon(xy, polygon):
    """Vektorisierter Punkt-im-Polygon-Test (Ray Casting).

    xy:      (N, 2) Bildschirmkoordinaten der Punkte
    polygon: (M, 2) Eckpunkte des Polygons
    """
    import numpy as np

    x = xy[:, 0]
    y = xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_intersect)
        j = i
    return inside


def crop_point_cloud(input_path, output_path, log_file):
    """Interaktives Zuschneiden einer Punktwolke mit frei gezeichnetem Polygon.

    Die Auswahl ist unabhaengig von den Picking-Funktionen von PyVista (deren
    API sich zwischen Versionen aendert): Die Polygon-Eckpunkte werden per
    Tastendruck an der Mausposition gesetzt, danach werden alle Punkte mit der
    Kameramatrix auf den Bildschirm projiziert und gegen das Polygon getestet.
    """
    import numpy as np
    import pyvista as pv
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D

    with open(input_path, "r", encoding="utf-8") as f:
        lines = (line.replace(",", " ") for line in f)
        points = np.loadtxt(
            lines,
            dtype=np.float64,
            usecols=(0, 1, 2),
            comments=("X", "x", "#"),
            ndmin=2,
        )

    if points.shape[0] == 0:
        raise ValueError(f"No points found in input cloud: {input_path}")

    # Offset abziehen gegen Ruckeln bei grossen Koordinaten
    offset = points.mean(axis=0)
    local_points = points - offset

    cloud = pv.PolyData(local_points)
    # Z-Koordinaten als Farbe (erzeugt automatisch die Colourbar)
    cloud["Hoehe (Z)"] = points[:, 2]

    plotter = pv.Plotter(title=f"Crop: {input_path.name}")
    plotter.add_points(
        cloud,
        scalars="Hoehe (Z)",
        cmap="terrain",
        point_size=3,
        render_points_as_spheres=False,
        scalar_bar_args = {
            "title": "Höhe Z [m]",
            "fmt": "%.1f",
        }
    )
    plotter.add_text(
        "Polygon-Auswahl:\n"
        "  A       = Eckpunkt an Mausposition setzen\n"
        "  Z       = letzten Eckpunkt entfernen\n"
        "  N       = Polygon neu beginnen\n"
        "  Enter/X = Auswahl bestaetigen\n"
        "Danach Fenster schliessen.",
        position="upper_left",
        font_size=10,
    )

    vertices = []        # Polygon-Eckpunkte in Bildschirmpixeln
    mouse_pos = [0, 0]   # zuletzt bekannte Mausposition
    selected_points = []

    # --- 2D-Overlay zum Anzeigen des Polygons -------------------------------
    overlay_mapper = vtkPolyDataMapper2D()
    overlay_actor = vtkActor2D()
    overlay_actor.SetMapper(overlay_mapper)
    overlay_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    overlay_actor.GetProperty().SetLineWidth(2.0)
    overlay_actor.GetProperty().SetPointSize(8.0)
    try:
        plotter.add_actor(
            overlay_actor,
            reset_camera=False,
            pickable=False,
            render=False,
        )
    except Exception:
        # Fallback für PyVista-Versionen, in denen add_actor 2D-Actors ablehnt
        plotter.renderer.AddViewProp(overlay_actor)

    def update_overlay():
        n = len(vertices)
        vtk_points = vtkPoints()
        for x, y in vertices:
            vtk_points.InsertNextPoint(x, y, 0.0)

        verts = vtkCellArray()
        for i in range(n):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)

        lines = vtkCellArray()
        if n >= 2:
            closed = n >= 3
            lines.InsertNextCell(n + (1 if closed else 0))
            for i in range(n):
                lines.InsertCellPoint(i)
            if closed:
                lines.InsertCellPoint(0)

        poly = vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetVerts(verts)
        poly.SetLines(lines)
        overlay_mapper.SetInputData(poly)
        plotter.render()

    # --- Projektion der Punktwolke auf den Bildschirm ------------------------
    def project_to_screen():
        width, height = plotter.render_window.GetSize()
        matrix = plotter.camera.GetCompositeProjectionTransformMatrix(
            width / height, -1, 1
        )
        m = np.array(
            [[matrix.GetElement(i, j) for j in range(4)] for i in range(4)]
        )
        clip_xyz = local_points @ m[:3, :3].T + m[:3, 3]
        w = local_points @ m[3, :3] + m[3, 3]
        visible = w > 0  # nur Punkte vor der Kamera
        w_safe = np.where(visible, w, 1.0)
        ndc = clip_xyz / w_safe[:, None]
        screen = np.column_stack(
            (
                (ndc[:, 0] + 1.0) * 0.5 * width,
                (ndc[:, 1] + 1.0) * 0.5 * height,
            )
        )
        return screen, visible

    # --- Tastenbelegung -------------------------------------------------------
    def on_mouse_move(obj, _event):
        mouse_pos[:] = obj.GetEventPosition()

    def add_vertex():
        vertices.append((float(mouse_pos[0]), float(mouse_pos[1])))
        update_overlay()

    def remove_vertex():
        if vertices:
            vertices.pop()
            update_overlay()

    def reset_polygon():
        vertices.clear()
        update_overlay()

    def confirm_selection():
        if len(vertices) < 3:
            print("\nPlease set at least 3 vertices first (Key 'A').", flush=True)
            return
        screen, visible = project_to_screen()
        inside = visible & _points_in_polygon(
            screen, np.asarray(vertices, dtype=np.float64)
        )
        if not inside.any():
            print("\nKeine Punkte im Polygon. Bitte erneut versuchen.", flush=True)
            return
        selected_points.clear()
        selected_points.append(points[inside])
        print(
            f"\nSuccess: {int(inside.sum())} Points selected! "
            "You can now close the window "
            "(or use N / A to set a new polygon).",
            flush=True,
        )

    plotter.iren.interactor.AddObserver("MouseMoveEvent", on_mouse_move)
    plotter.add_key_event("a", add_vertex)
    plotter.add_key_event("z", remove_vertex)
    plotter.add_key_event("n", reset_polygon)
    plotter.add_key_event("Return", confirm_selection)
    plotter.add_key_event("x", confirm_selection)
    update_overlay()

    _print_and_log(
        f"Cropping {input_path.name}: Rotate the model into position, set the polygon vertices with "
        "the 'A' key at the mouse position, confirm with "
        "Enter (or X) and close the window.",
        log_file,
    )

    plotter.show()

    if not selected_points:
        raise ValueError(f"Error: No points selected for {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, selected_points[0], fmt="%.6f")

    _print_and_log(
        f"Saved cropped point cloud ({len(selected_points[0])} points) to {output_path}.",
        log_file,
    )


def crop_input_clouds(data_path, data_set, res_root, run_timestamp, log_file):
    crop_dir = res_root / data_set / f"cropped_{run_timestamp}"
    if crop_dir.exists():
        raise FileExistsError(f"Cropped-cloud output directory already exists: {crop_dir}")

    if data_path.is_file():
        source_files = [data_path]
    else:
        source_files = sorted(
            (
                item
                for item in data_path.iterdir()
                if item.is_file()
                and item.suffix.lower() == ".xyz"
                and not item.stem.lower().endswith("_gt")
            ),
            key=lambda item: item.name.casefold(),
        )

    for source_file in source_files:
        crop_point_cloud(
            source_file,
            crop_dir / f"{source_file.stem}.xyz",
            log_file,
        )

    return (
        crop_dir / f"{source_files[0].stem}.xyz"
        if data_path.is_file()
        else crop_dir
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run normal estimation, followed by rock discontinuity extraction."
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to one .xyz point cloud or a directory of .xyz point clouds.",
    )
    parser.add_argument(
        "--cut_cloud",
        "--cut-cloud",
        dest="cut_cloud",
        type=parse_bool,
        default=False,
        help="Interactively crop each input cloud before analysis.",
    )
    parser.add_argument(
        "--log-root",
        default=str(PROJECT_ROOT / "logs"),
        help="Root directory for logs from both stages.",
    )

    # Normal-estimation options
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
    parser.add_argument("--rock-show-fitted-planes", type=parse_bool, default=True)
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
    if args.cut_cloud and args.norm_evaluate:
        parser.error(
            "--norm-evaluate cannot be used with --cut_cloud true because "
            "ground-truth normals are not cropped with the point cloud."
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

def shift_clouds_to_local(data_path, work_dir, shape_names):
    import numpy as np
    shifted_dir = work_dir / "shifted_clouds"
    shifted_dir.mkdir(parents=True, exist_ok=True)
    offsets = {}
    
    source_dir = data_path if data_path.is_dir() else data_path.parent
    files_to_process = [data_path] if data_path.is_file() else [source_dir / f"{name}.xyz" for name in shape_names]
    
    for file_path in files_to_process:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = (line.replace(",", " ") for line in f)
            data = np.loadtxt(lines, dtype=np.float64, comments=("X", "x", "#"), ndmin=2)
        
        # Offset berechnen und abziehen (nur von Spalte 0, 1, 2)
        offset = data[:, :3].mean(axis=0)
        data[:, :3] -= offset
        offsets[file_path.stem] = offset
        
        out_path = shifted_dir / file_path.name
        np.savetxt(out_path, data, fmt="%.6f")
        
    new_data_path = shifted_dir / files_to_process[0].name if data_path.is_file() else shifted_dir
    return new_data_path, offsets


def restore_clouds_to_global(res_root, data_set, offsets):
    import numpy as np
    orientation_root = res_root / data_set
    
    # Alle .xyz-Dateien in den Ergebnissen suchen (Cluster, Planes etc.)
    for xyz_file in orientation_root.rglob("*.xyz"):
        matching_shape = next((shape for shape in offsets if xyz_file.name.startswith(shape)), None)
        
        if matching_shape:
            data = np.loadtxt(xyz_file, dtype=np.float64, ndmin=2)
            # Offset wieder auf X, Y, Z addieren
            data[:, :3] += offsets[matching_shape]
            np.savetxt(xyz_file, data, fmt="%.6f")



def run_pipeline(args):
    log_root = Path(args.log_root).resolve()
    res_root = Path(args.rock_res_root).resolve()
    data_path, data_set, shape_names = resolve_data_path(args.data_path)

    if not NORMAL_MAIN.is_file():
        raise FileNotFoundError(f"Normal-estimation script not found: {NORMAL_MAIN}")
    if not ROCK_MAIN.is_file():
        raise FileNotFoundError(f"Rock-extraction script not found: {ROCK_MAIN}")
    if args.rock_save_orientation_res and not STEREOPLOT_MAIN.is_file():
        raise FileNotFoundError(f"Stereoplot script not found: {STEREOPLOT_MAIN}")

    log_root.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    combined_log = log_root / f"{data_set}_{run_timestamp}.txt"

    with (
        tempfile.TemporaryDirectory(prefix="discontinuity-analysis-") as temp_dir,
        combined_log.open("x", encoding="utf-8") as run_log,
    ):
        if args.cut_cloud:
            data_path = crop_input_clouds(
                data_path,
                data_set,
                res_root,
                run_timestamp,
                run_log,
            )

        #--- Global Shift anwenden ---
        _print_and_log("Applying Global Shift to center coordinates...", run_log)
        data_path, offsets = shift_clouds_to_local(data_path, Path(temp_dir), shape_names)
        # ----------------------------------

        normal_data_set = data_path.stem if data_path.is_file() else data_path.name
        input_batch = "single" if data_path.is_file() else "files"

        normal_data_set = data_path.stem if data_path.is_file() else data_path.name
        input_batch = "single" if data_path.is_file() else "files"
        work_root = Path(temp_dir)
        normal_log_root = work_root / "normal-logs"
        normal_log_parent = normal_log_root / normal_data_set / input_batch
        normal_log_parent.mkdir(parents=True, exist_ok=True)
        previous_runs = {
            path
            for path in normal_log_parent.iterdir()
            if path.is_dir()
        }

        normal_command = [
            sys.executable,
            "-u",
            str(NORMAL_MAIN),
            "--data_path", str(data_path),
            "--log_root", str(normal_log_root),
            "--evaluate", str(args.norm_evaluate),
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

        _print_and_log("Running normal-vector estimation...", run_log)
        _run_and_log(
            normal_command,
            NORMAL_MAIN.parent,
            run_log,
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
        staged_root = work_root / "rock-extraction"
        orientation_root = res_root / data_set

        for shape_name in shape_names:
            prediction_file = prediction_dir / f"{shape_name}.xyz"
            if not prediction_file.is_file():
                raise FileNotFoundError(
                    f"Expected prediction was not created: {prediction_file}"
                )

            staged_dataset_dir = staged_root / data_set
            staged_dataset_dir.mkdir(parents=True, exist_ok=True)

            staged_input = (
                staged_dataset_dir
                / f"{data_set}{args.rock_data_suffix}"
            )
            shutil.copy2(prediction_file, staged_input)

            rock_command = [
                sys.executable,
                "-u",
                str(ROCK_MAIN),
                "--dataset_root", str(staged_root),
                "--data_set", data_set,
                "--res_root", str(res_root),
                "--log_root", str(normal_log_root),
                "--data_suffix", args.rock_data_suffix,
                "--file_suffix", args.rock_file_suffix,
                "--angle", str(args.rock_angle),
                "--iter", str(args.rock_iter),
                "--min_clusters", str(args.rock_min_clusters),
                "--show_cluster_res", str(args.rock_show_cluster_res),
                "--save_cluster_res", str(args.rock_save_cluster_res),
                "--show_planes_res", str(args.rock_show_planes_res),
                "--show_fitted_planes_res", str(args.rock_show_fitted_planes),
                "--save_planes_res", str(args.rock_save_planes_res),
                "--save_orientation_res", str(args.rock_save_orientation_res),
            ]

            previous_orientation_times = {}
            if args.rock_save_orientation_res and orientation_root.is_dir():
                previous_orientation_times = {
                    path: path.stat().st_mtime_ns
                    for path in orientation_root.rglob(
                        f"orientations{args.rock_file_suffix}"
                    )
                    if path.is_file()
                }

            _print_and_log(f"Running rock extraction for {shape_name}...", run_log)
            _run_and_log(
                rock_command,
                ROCK_MAIN.parent,
                run_log,
            )

            if args.rock_save_orientation_res:
                updated_orientation_files = [
                    path
                    for path in orientation_root.rglob(
                        f"orientations{args.rock_file_suffix}"
                    )
                    if path.is_file()
                    and previous_orientation_times.get(path) != path.stat().st_mtime_ns
                ]
                if len(updated_orientation_files) != 1:
                    raise RuntimeError(
                        "Could not identify the orientation output for "
                        f"{shape_name}. Expected one updated file, found "
                        f"{len(updated_orientation_files)}."
                    )

                _run_and_log(
                    [
                        sys.executable,
                        "-u",
                        str(STEREOPLOT_MAIN),
                        str(updated_orientation_files[0]),
                        "--title",
                        shape_name,
                    ],
                    STEREOPLOT_MAIN.parent,
                    run_log,
                )
            else:
                _print_and_log(
                    "Stereoplot skipped because "
                    "--rock-save-orientation-res is false.",
                    run_log,
                )

        # --- NEU: Global Shift wieder auf die Ergebnisse addieren ---
        _print_and_log("Restoring original global coordinates...", run_log)
        restore_clouds_to_global(res_root, data_set, offsets)
        # ------------------------------------------------------------
        _print_and_log("Both stages completed successfully.", run_log)


if __name__ == "__main__":
    run_pipeline(parse_arguments())