import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

from typing import Any, List, Tuple, Optional, Union, Iterator, Dict
from pathlib import Path
import contextlib
from argparse import ArgumentParser
from joblib import Parallel, delayed
import math
from time import time
import logging
import tracemalloc
import linecache

from tqdm import tqdm
import trimesh
from trimesh import Trimesh
import numpy as np
import mcubes
import pyrender
import pymeshlab
from scipy import ndimage
from scipy.spatial.transform import Rotation

import libfusiongpu as libfusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_points(n_views: int = 100) -> np.ndarray:
    """See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere."""
    rnd = 1.
    points = []
    offset = 2. / n_views
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)


def get_views(points: np.ndarray) -> List[np.ndarray]:
    """Generate a set of views to generate depth maps from."""
    Rs = []
    for i in range(points.shape[0]):
        # https://np.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array(
            [[1, 0, 0], [0, math.cos(latitude), -math.sin(latitude)], [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array(
            [[math.cos(longitude), 0, math.sin(longitude)], [0, 1, 0], [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)
        Rs.append(R)

    return Rs


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = Parallel.print_progress
    Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        Parallel.print_progress = original_print_progress
        tqdm_object.close()


def resolve_out_path(in_path: Path,
                     in_dir: Path,
                     out_format: str,
                     out_dir: Optional[Path] = None) -> Path:
    in_path = in_path.expanduser().resolve()
    if out_dir is not None:
        in_dir = in_dir.expanduser().resolve()
        out_dir = out_dir.expanduser().resolve()
        return out_dir / in_path.relative_to(in_dir).with_suffix(out_format)
    return in_path.with_suffix(out_format)


def load(in_path: Path,
         loader: str = "pymeshlab",
         return_type: str = "dict") -> Union[Trimesh, Dict[str, np.ndarray]]:
    if loader == "trimesh":
        mesh = trimesh.load(in_path,
                            force="mesh",
                            process=False,
                            validate=False)
        vertices = mesh.vertices
        faces = mesh.faces
    elif loader == "pymeshlab":
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(in_path))
        vertices = ms.current_mesh().vertex_matrix()
        faces = ms.current_mesh().face_matrix()
    else:
        raise ValueError(f"Unknown loader: {loader}.")

    if return_type == "dict":
        return {"vertices": vertices,
                "faces": faces}
    elif return_type == "trimesh":
        return Trimesh(vertices=vertices,
                       faces=faces,
                       process=False,
                       validate=False)
    else:
        raise ValueError(f"Unknown return type '{return_type}'.")


def normalize(mesh: Union[Trimesh, Dict[str, np.ndarray]],
              translation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
              scale: Optional[Union[float, Tuple[float, float, float], np.ndarray]] = None,
              padding: float = 0) -> Tuple[Union[Trimesh, Dict[str, np.ndarray]], np.ndarray, float]:
    if isinstance(mesh, Trimesh):
        if translation is None:
            translation = -mesh.bounds.mean(axis=0)
        if scale is None:
            max_extents = mesh.extents.max()
            # scale = 1 / (max_extents + 2 * padding * max_extents)
            scale = (1 - padding) / max_extents

        mesh.apply_translation(translation)
        mesh.apply_scale(scale)
    elif isinstance(mesh, dict):
        if translation is None or scale is None:
            vertices = mesh["vertices"]
            faces = mesh["faces"]
            referenced = np.zeros(len(vertices), dtype=bool)
            referenced[faces] = True
            in_mesh = vertices[referenced]
            bounds = np.array([in_mesh.min(axis=0), in_mesh.max(axis=0)])
            if translation is None:
                translation = -bounds.mean(axis=0)
            if scale is None:
                extents = bounds.ptp(axis=0)
                max_extents = extents.max()
                # scale = 1 / (max_extents + 2 * padding * max_extents)
                scale = (1 - padding) / max_extents

        mesh["vertices"] += translation
        mesh["vertices"] *= scale
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

    return mesh, translation, scale


def render(mesh: Union[Trimesh, Dict[str, np.ndarray]],
           rotations: List[np.ndarray],
           resolution: int,
           width: int,
           height: int,
           fx: float,
           fy: float,
           cx: float,
           cy: float,
           znear: float,
           zfar: float,
           offset: float = 0,
           erode: bool = True) -> List[np.ndarray]:
    renderer = pyrender.OffscreenRenderer(width, height)
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear, zfar)
    rot_x_180 = Rotation.from_euler('x', 180, degrees=True).as_matrix()

    depth_maps = list()
    for R in rotations:
        R_pyrender = rot_x_180 @ R

        if isinstance(mesh, Trimesh):
            trafo = np.eye(4)
            trafo[:3, :3] = R_pyrender
            trafo[:3, 3] = np.array([0, 0, -1])

            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(trafo)
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_copy)
        elif isinstance(mesh, dict):
            vertices = mesh["vertices"].copy()
            vertices = vertices @ R_pyrender.T
            vertices[:, 2] -= 1
            faces = mesh["faces"].copy()

            primitives = [pyrender.Primitive(positions=vertices, indices=faces)]
            pyrender_mesh = pyrender.Mesh(primitives=primitives)
        else:
            raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

        scene = pyrender.Scene()
        scene.add(pyrender_mesh)
        scene.add(camera)

        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        depth[depth == 0] = zfar

        # Optionally thicken object by offsetting and eroding the depth maps.
        depth -= offset * (1 / resolution)
        if erode:
            depth = ndimage.grey_erosion(depth, size=(3, 3))
        depth_maps.append(depth)

    renderer.delete()
    return depth_maps


def fuse(depth_maps: List[np.ndarray],
         rotations: List[np.ndarray],
         resolution: int,
         fx: float,
         fy: float,
         cx: float,
         cy: float) -> np.ndarray:
    Ks = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]]).reshape((1, 3, 3))

    Ks = np.repeat(Ks, len(depth_maps), axis=0).astype(np.float32)
    Rs = np.array(rotations).astype(np.float32)
    Ts = np.array([np.array([0, 0, 1]) for _ in range(len(Rs))]).astype(np.float32)
    depth_maps = np.array(depth_maps).astype(np.float32)
    voxel_size = 1 / resolution

    views = libfusion.PyViews(depth_maps, Ks, Rs, Ts)
    tsdf = libfusion.tsdf_gpu(views,
                              resolution,
                              resolution,
                              resolution,
                              voxel_size,
                              10 * voxel_size,
                              False)[0].transpose((2, 1, 0))
    return tsdf


def extract(tsdf: np.ndarray,
            resolution: int,
            return_type: str = "dict") -> Union[Trimesh, Dict[str, np.ndarray]]:
    tsdf = np.pad(tsdf, 1, "constant", constant_values=1e6)
    vertices, triangles = mcubes.marching_cubes(-tsdf, 0)

    vertices -= 1
    vertices /= resolution
    vertices -= 0.5

    if return_type == "trimesh":
        return Trimesh(vertices=vertices,
                       faces=triangles,
                       process=False,
                       validate=False)
    elif return_type == "dict":
        return {"vertices": vertices, "faces": triangles}
    else:
        raise ValueError(f"Unknown return type '{return_type}'.")


def process(mesh: Union[Trimesh, Dict[str, np.ndarray]],
            script_paths: Iterator[Path]) -> Union[Trimesh, Dict[str, np.ndarray]]:
    if isinstance(mesh, Trimesh):
        vertices = mesh.vertices
        faces = mesh.faces
    elif isinstance(mesh, dict):
        vertices = mesh["vertices"]
        faces = mesh["faces"]
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

    ms = pymeshlab.MeshSet()
    pymesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(pymesh)

    for script_path in script_paths:
        logger.debug(f"\tprocess: Applying script {script_path}.")
        ms.load_filter_script(str(script_path))
        ms.apply_filter_script()

    if isinstance(mesh, Trimesh):
        return Trimesh(vertices=ms.current_mesh().vertex_matrix(),
                       faces=ms.current_mesh().face_matrix(),
                       process=False,
                       validate=False)
    elif isinstance(mesh, dict):
        return {"vertices": ms.current_mesh().vertex_matrix(),
                "faces": ms.current_mesh().face_matrix()}


def save(mesh: Union[Trimesh, pymeshlab.MeshSet, pymeshlab.Mesh, Dict[str, np.ndarray]],
         path: Path,
         precision: int = 32):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(mesh, Trimesh):
        if precision == 16:
            precision = np.float16
        elif precision == 32:
            precision = np.float32
        elif precision == 64:
            precision = np.float64
        else:
            raise ValueError(f"Invalid precision: {precision}.")
        precision = np.finfo(precision).precision
        mesh.export(path, digits=precision)
    elif isinstance(mesh, (pymeshlab.MeshSet, pymeshlab.Mesh, dict)):
        if isinstance(mesh, pymeshlab.Mesh):
            ms = pymeshlab.MeshSet()
            ms.add_mesh(mesh)
        elif isinstance(mesh, dict):
            ms = pymeshlab.MeshSet()
            pymesh = pymeshlab.Mesh(vertex_matrix=mesh["vertices"], face_matrix=mesh["faces"])
            ms.add_mesh(pymesh)
        else:
            ms = mesh
        ms.save_current_mesh(file_name=str(path),
                             save_vertex_color=False,
                             save_vertex_coord=False,
                             save_face_color=False,
                             save_polygonal=True)
    else:
        raise ValueError(f"Unsupported mesh type '{type(mesh)}'.")


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def trace_run(in_path: Path, args: Any):
    snapshot1 = tracemalloc.take_snapshot()

    run(in_path, args)

    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("[ Top 10 differences ]")
    for stat in top_stats[:10]:
        print(stat)

    display_top(snapshot2)


def run(in_path: Path, args: Any):
    start = time()
    logger.debug(f"Processing file {in_path}:")

    out_path = resolve_out_path(in_path, args.in_dir, args.out_format, args.out_dir)
    if out_path.exists() and not args.overwrite:
        logger.debug(f"File {out_path} already exists. Skipping.")
        return

    try:
        restart = time()
        mesh = load(in_path,
                    loader="trimesh" if args.use_trimesh else "pymeshlab",
                    return_type="trimesh" if args.use_trimesh else "dict")
        if args.use_trimesh:
            vertices = mesh.vertices
        else:
            vertices = mesh["vertices"]
        logger.debug(f"Loaded mesh ({len(vertices)} vertices) in {time() - restart:.2f}s.")

        restart = time()
        mesh, translation, scale = normalize(mesh, padding=args.padding)
        logger.debug(f"Normalized mesh in {time() - restart:.2f}s.")

        restart = time()
        rotations = get_views(get_points(args.n_views))
        logger.debug(f"Generated rotations in {time() - restart:.2f}s.")

        restart = time()
        depth_maps = render(mesh,
                            rotations,
                            args.resolution,
                            args.width,
                            args.height,
                            args.fx,
                            args.fy,
                            args.cx,
                            args.cy,
                            args.znear,
                            args.zfar,
                            args.depth_offset,
                            args.erode)
        logger.debug(f"Rendered depth maps in {time() - restart:.2f}s.")

        restart = time()
        tsdf = fuse(depth_maps,
                    rotations,
                    args.resolution,
                    args.fx,
                    args.fy,
                    args.cx,
                    args.cy)
        logger.debug(f"Fused depth maps in {time() - restart:.2f}s.")

        restart = time()
        mesh = extract(tsdf, args.resolution, return_type="trimesh" if args.use_trimesh else "dict")
        if args.use_trimesh:
            vertices = mesh.vertices
        else:
            vertices = mesh["vertices"]
        logger.debug(f"Extracted mesh ({len(vertices)} vertices) in {time() - restart:.2f}s.")

        if args.script_dir:
            restart = time()
            mesh = process(mesh, sorted(args.script_dir.expanduser().resolve().glob("*.mlx")))
            if args.use_trimesh:
                vertices = mesh.vertices
            else:
                vertices = mesh["vertices"]
            logger.debug(f"Filtered mesh ({len(vertices)} vertices) in {time() - restart:.2f}s.")

        restart = time()
        mesh, _, _ = normalize(mesh, translation=-translation * 1 / scale, scale=1 / scale)
        logger.debug(f"Normalized mesh in {time() - restart:.2f}s.")

        restart = time()
        save(mesh, out_path, args.precision)
        logger.debug(f"Saved mesh in {time() - restart:.2f}s.")
    except Exception as e:
        logger.exception(e)
    logger.debug(f"Runtime: {time() - start:.2f}s.\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=Path, required=True, help="Path to input directory.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--in_format", type=str, default=".obj", help="Input file format.")
    parser.add_argument("--out_format", type=str, default=".off", help="Output file format.")
    parser.add_argument("--script_dir", type=Path, help="Path to directory containing MeshLab scripts.")
    parser.add_argument("--width", type=int, default=640, help="Width of the depth map.")
    parser.add_argument("--height", type=int, default=480, help="Height of the depth map.")
    parser.add_argument("--fx", type=float, default=640, help="Focal length in x.")
    parser.add_argument("--fy", type=float, default=640, help="Focal length in y.")
    parser.add_argument("--cx", type=float, default=320, help="Principal point in x.")
    parser.add_argument("--cy", type=float, default=240, help="Principal point in y.")
    parser.add_argument("--znear", type=float, default=0.25, help="Near clipping plane.")
    parser.add_argument("--zfar", type=float, default=1.75, help="Far clipping plane.")
    parser.add_argument("--padding", type=float, default=0.1, help="Relative padding applied on each side.")
    parser.add_argument("--resolution", type=int, default=255, help="Resolution of the TSDF fusion voxel grid.")
    parser.add_argument("--depth_offset", type=float, default=0,
                        help="Thicken object through offsetting of rendered depth maps.")
    parser.add_argument("--erode", action="store_true", help="Erode rendered depth maps to thicken thin structures.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs.")
    parser.add_argument("--n_views", type=int, default=100, help="Number of views to render.")
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32, 64], help="Data precision.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--use_trimesh", action="store_true", help="Use trimesh for loading and saving meshes.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.use_trimesh:
        logging.getLogger("trimesh").setLevel(logging.ERROR)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    in_dir = args.in_dir.expanduser().resolve()
    logger.debug(f"Globbing paths from {in_dir}.")
    files = sorted(in_dir.rglob(f"*{args.in_format}"))

    # tracemalloc.start()

    progress = tqdm(desc="Mesh Fusion", total=len(files), disable=args.verbose)
    with tqdm_joblib(progress):
        Parallel(n_jobs=args.n_jobs)(delayed(run)(file, args) for file in files)


if __name__ == "__main__":
    main()
