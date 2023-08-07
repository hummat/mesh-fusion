import gc
import os
from random import shuffle
import tempfile
from functools import partial

os.environ["PYOPENGL_PLATFORM"] = "egl"

from typing import Any, List, Tuple, Optional, Union, Iterator, Dict
from pathlib import Path
import contextlib
from argparse import ArgumentParser
from joblib import Parallel, delayed
import math
from time import time, sleep
import logging
import tracemalloc
import linecache

import pymeshlab
from pykdtree.kdtree import KDTree
from tqdm import tqdm
import trimesh
from trimesh import Trimesh
import numpy as np
import mcubes
import pyrender
from scipy import ndimage
from scipy.spatial.transform import Rotation
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODES = ["fuse", "carve", "fill"]
try:
    import libfusiongpu as libfusion
except ImportError:
    logger.warning("Could not import libfusiongpu, falling back to CPU implementation.")
    try:
        import libfusioncpu as libfusion
    except ImportError:
        logger.warning("Could not import libfusioncpu, 'fuse' mode disabled.")
        MODES.remove("fuse")
try:
    import open3d as o3d
except ImportError:
    logger.warning("Could not import Open3D, 'carve' mode disabled.")
    MODES.remove("carve")
try:
    import torch
    import kaolin
except ImportError:
    logger.warning("Could not import PyTorch and/or NVIDIA Kaolin, 'fill' mode disabled.")
    MODES.remove("fill")
assert len(MODES) > 0, "No modes available, exiting."
logger.debug(f"Enabled modes: {MODES}")


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
              padding: float = 0.1) -> Tuple[Union[Trimesh, Dict[str, np.ndarray]], np.ndarray, float]:
    if isinstance(mesh, Trimesh):
        if translation is None:
            translation = -mesh.bounds.mean(axis=0)
        if scale is None:
            max_extents = mesh.extents.max()
            scale = 1 / (max_extents + 2 * padding * max_extents)
            # scale = (1 - padding) / max_extents

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
                scale = 1 / (max_extents + 2 * padding * max_extents)
                # scale = (1 - padding) / max_extents

        mesh["vertices"] += translation
        mesh["vertices"] *= scale
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

    return mesh, translation, scale


def voxelize(mesh: Union[Trimesh, Dict[str, np.ndarray]],
             min_value: float = -0.5,
             max_value: float = 0.5,
             resolution: int = 256) -> np.ndarray:
    x = np.linspace(min_value, max_value, resolution)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    grid_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

    vertices, faces = get_vertices_and_faces(mesh)
    points = Trimesh(vertices=vertices, faces=faces).sample(resolution ** 3)

    _, indices = KDTree(grid_points, leafsize=100).query(points)
    voxel = np.zeros(len(grid_points), dtype=bool)
    voxel[indices] = True

    return voxel.reshape((resolution,) * 3)


def kaolin_pipeline(mesh: Union[Trimesh, Dict[str, np.ndarray]],
                    resolution: int = 256,
                    fill_holes: bool = True,
                    eps: float = 1e-6,
                    smoothing_iterations: int = 3,
                    realign: bool = True,
                    save_voxel_path: Optional[Path] = None,
                    try_cpu: bool = False) -> Dict[str, np.ndarray]:
    vertices, faces = get_vertices_and_faces(mesh)
    torch_vertices = torch.from_numpy(vertices).cuda()
    torch_faces = torch.from_numpy(faces).cuda()

    try:
        voxel = kaolin.ops.conversions.trianglemeshes_to_voxelgrids(torch_vertices.unsqueeze(0),
                                                                    torch_faces,
                                                                    resolution=resolution,
                                                                    origin=torch.zeros((1, 3)).cuda() - 0.5,
                                                                    scale=torch.ones(1).cuda())
    except torch.cuda.OutOfMemoryError as e:
        if try_cpu:
            logger.error("Out of memory error during voxelization on GPU. Trying CPU implementation.")
            voxel = torch.from_numpy(voxelize(mesh, resolution=resolution)).unsqueeze(0).cuda()
        else:
            raise e

    if fill_holes:
        voxel = torch.from_numpy(ndimage.binary_fill_holes(voxel.squeeze(0).cpu().numpy())).unsqueeze(0).cuda()
    voxel = kaolin.ops.voxelgrid.extract_surface(voxel)
    odms = kaolin.ops.voxelgrid.extract_odms(voxel)
    voxel = kaolin.ops.voxelgrid.project_odms(odms)

    try:
        vertices, faces = kaolin.ops.conversions.voxelgrids_to_trianglemeshes(voxel)
        vertices = vertices[0] / (voxel.size(-1) + 1)
        vertices -= 0.5
        faces = faces[0]
    except torch.cuda.OutOfMemoryError as e:
        if try_cpu:
            logger.error("Out of memory error during Marching Cubes on GPU. Trying CPU implementation.")
            mesh = extract(voxel.squeeze(0).cpu().numpy(), level=0.5, resolution=resolution, pad=False)
            vertices, faces = get_vertices_and_faces(mesh)
            vertices = torch.from_numpy(vertices).cuda()
            faces = torch.from_numpy(faces).cuda()
        else:
            raise e

    voxel = voxel.squeeze(0).cpu().numpy()
    if save_voxel_path is not None:
        save_voxel_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(save_voxel_path), voxel=np.packbits(voxel))

    if smoothing_iterations > 0:
        adj_sparse = kaolin.ops.mesh.adjacency_matrix(len(vertices), faces, sparse=True)
        neighbor_num = torch.sparse.sum(adj_sparse, dim=1).to_dense().view(-1, 1)
        for _ in range(smoothing_iterations):
            neighbor_sum = torch.sparse.mm(adj_sparse, vertices)
            vertices = neighbor_sum / neighbor_num

    if realign:
        src_min, src_max = vertices.min(0, keepdim=True)[0], vertices.max(0, keepdim=True)[0]
        tgt_min, tgt_max = torch_vertices.min(0, keepdim=True)[0], torch_vertices.max(0, keepdim=True)[0]
        vertices = ((vertices - src_min) / (src_max - src_min + eps)) * (tgt_max - tgt_min) + tgt_min

    return {"vertices": vertices.cpu().numpy(), "faces": faces.cpu().numpy()}


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
           offset: float = 1.5,
           erode: bool = True,
           flip_faces: bool = False,
           show: bool = False) -> List[np.ndarray]:
    renderer = pyrender.OffscreenRenderer(width, height)
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear, zfar)
    rot_x_180 = Rotation.from_euler('x', 180, degrees=True).as_matrix()

    depthmaps = list()
    for R in rotations:
        R_pyrender = rot_x_180 @ R

        if isinstance(mesh, Trimesh):
            trafo = np.eye(4)
            trafo[:3, :3] = R_pyrender
            trafo[:3, 3] = np.array([0, 0, -1])

            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(trafo)

            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_copy)

            if flip_faces:
                mesh_copy.invert()
                pyrender_mesh = [pyrender_mesh, pyrender.Mesh.from_trimesh(mesh_copy)]
        elif isinstance(mesh, dict):
            vertices = mesh["vertices"].copy()
            vertices = vertices @ R_pyrender.T
            vertices[:, 2] -= 1
            faces = mesh["faces"].copy()

            primitives = [pyrender.Primitive(positions=vertices, indices=faces)]

            if flip_faces:
                primitives.append(pyrender.Primitive(positions=vertices, indices=np.flip(faces, axis=1)))

            pyrender_mesh = pyrender.Mesh(primitives=primitives)
        else:
            raise ValueError(f"Unknown mesh type '{type(mesh)}'.")

        scene = pyrender.Scene()
        scene.add(pyrender_mesh)
        scene.add(camera)

        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)

        if show:
            Image.fromarray(((depth / depth.max()) * 256).astype(np.uint8)).convert('L').show()
            sleep(1)

        depth[depth == 0] = zfar

        # Optionally thicken object by offsetting and eroding the depth maps.
        depth -= offset * (1 / resolution)
        if erode:
            depth = ndimage.grey_erosion(depth, size=(3, 3))
        depthmaps.append(depth)

    renderer.delete()
    return depthmaps


def fuse(depthmaps: List[np.ndarray],
         rotations: List[np.ndarray],
         resolution: int,
         fx: float,
         fy: float,
         cx: float,
         cy: float) -> np.ndarray:
    Ks = np.array([[fx, 0, cx],
                   [0, fy, cy],
                   [0, 0, 1]]).reshape((1, 3, 3))

    Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)
    Rs = np.array(rotations).astype(np.float32)
    Ts = np.array([np.array([0, 0, 1]) for _ in range(len(Rs))]).astype(np.float32)
    depthmaps = np.array(depthmaps).astype(np.float32)
    voxel_size = 1 / resolution

    views = libfusion.PyViews(depthmaps, Ks, Rs, Ts)
    tsdf = libfusion.tsdf_gpu(views,
                              depth=resolution,
                              height=resolution,
                              width=resolution,
                              vx_size=voxel_size,
                              truncation=10 * voxel_size,
                              unknown_is_free=False)[0].transpose((2, 1, 0))
    return tsdf


def extract(grid: np.ndarray,
            level: float,
            resolution: int,
            pad: bool = True,
            return_type: str = "dict") -> Union[Trimesh, Dict[str, np.ndarray]]:
    if pad:
        grid = np.pad(grid, 1, "constant", constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(grid, level)
    if pad:
        vertices -= 1
    vertices /= (resolution - 1)
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


def load_scripts(script_dir: Path,
                 num_vertices: Optional[int] = None,
                 min_vertices: int = 20000,
                 max_vertices: int = 200000) -> List[Path]:
    scripts = sorted(script_dir.glob("*.mlx"))
    logger.debug(f"Found {len(scripts)} scripts in {script_dir}.")
    simplify = any(script.name == 'simplify.mlx' for script in scripts)
    if simplify and num_vertices is not None:
        if num_vertices < min_vertices:
            percentage = 0.5
        elif num_vertices > max_vertices:
            percentage = 0.1
        else:
            percentage = round(0.5 + (num_vertices - min_vertices) / (max_vertices - min_vertices) * (0.1 - 0.5), 2)
        logger.debug(f"\tload_scripts: Simplifying mesh by {100 * (1 - percentage):.0f}%.")

        index = next(i for i, s in enumerate(scripts) if s.name == 'simplify.mlx')
        with open(scripts[index], 'r') as f:
            script = f.read()
            assert 'Simplification: Quadric Edge Collapse Decimation' in script
            script = script.replace('"Percentage reduction (0..1)" value="0.05"',
                                    f'"Percentage reduction (0..1)" value="{percentage}"')
            assert f'"Percentage reduction (0..1)" value="{percentage}"' in script

            scripts[index] = Path(tempfile.mkstemp(suffix=".mlx")[1])
            scripts[index].write_text(script)
            logger.debug(f"\tload_scripts: Saved modified simplification script to {scripts[index]}.")
    return scripts


def get_vertices_and_faces(mesh: Union[Trimesh, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, Trimesh):
        vertices, faces = mesh.vertices, mesh.faces
    elif isinstance(mesh, dict):
        vertices, faces = mesh["vertices"], mesh["faces"]
    else:
        raise ValueError(f"Unknown mesh type '{type(mesh)}'.")
    return vertices.astype(np.float32), faces.astype(np.int64)


def process(mesh: Union[Trimesh, Dict[str, np.ndarray]],
            script_paths: List[Path]) -> Union[Trimesh, Dict[str, np.ndarray]]:
    vertices, faces = get_vertices_and_faces(mesh)

    ms = pymeshlab.MeshSet()
    pymesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms.add_mesh(pymesh)

    for script_path in script_paths:
        logger.debug(f"\tprocess: Applying script {script_path.name}.")
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

        save_current_mesh = partial(ms.save_current_mesh, save_textures=False)
        if path.suffix == ".stl":
            save_current_mesh = partial(save_current_mesh,
                                        binary=True,
                                        save_face_color=False)
        if path.suffix in [".off", ".ply", ".obj"]:
            save_current_mesh = partial(save_current_mesh,
                                        save_vertex_color=False,
                                        save_vertex_coord=False,
                                        save_face_color=False,
                                        save_polygonal=False)
        if path.suffix == ".ply":
            save_current_mesh = partial(save_current_mesh,
                                        binary=True,
                                        save_vertex_quality=False,
                                        save_vertex_normal=False,
                                        save_vertex_radius=False,
                                        save_face_quality=False,
                                        save_wedge_color=False,
                                        save_wedge_texcoord=False,
                                        save_wedge_normal=False)
        elif path.suffix == ".obj":
            save_current_mesh = partial(save_current_mesh,
                                        save_vertex_normal=False,
                                        save_wedge_texcoord=False,
                                        save_wedge_normal=False)
        save_current_mesh(file_name=str(path))
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


def check(out_path: Path, args: Any):
    try:
        mesh = load(out_path,
                    loader="trimesh" if args.use_trimesh else "pymeshlab",
                    return_type="trimesh" if args.use_trimesh else "dict")
        vertices, faces = get_vertices_and_faces(mesh)
        assert len(vertices) > 0 and len(faces) > 0, f"Mesh {out_path} is empty."
        if args.check_watertight:
            mesh = Trimesh(vertices=vertices, faces=faces, process=False, validate=False)
            assert mesh.is_watertight, f"Mesh {out_path} is not watertight."
    except Exception as e:
        logger.exception(e)
        if args.fix:
            try:
                os.remove(out_path)
            except Exception as e:
                logger.exception(e)
                pass


def run(in_path: Path, args: Any):
    start = time()
    logger.debug(f"Processing file {in_path}:")

    out_path = resolve_out_path(in_path, args.in_dir, args.out_format, args.out_dir)
    if args.check:
        check(out_path, args)
        if not args.fix:
            return

    if out_path.exists() and not args.overwrite:
        logger.debug(f"File {out_path} already exists. Skipping.")
        return

    try:
        restart = time()
        mesh = load(in_path,
                    loader="trimesh" if args.use_trimesh else "pymeshlab",
                    return_type="trimesh" if args.use_trimesh else "dict")
        vertices = get_vertices_and_faces(mesh)[0]
        logger.debug(f"Loaded mesh ({len(vertices)} vertices) in {time() - restart:.2f}s.")

        translation = np.zeros(3)
        scale = 1.0
        if not args.no_normalization:
            restart = time()
            mesh, translation, scale = normalize(mesh, padding=args.padding)
            logger.debug(f"Normalized mesh in {time() - restart:.2f}s.")

        if args.mode == "fill":
            restart = time()
            mesh = kaolin_pipeline(mesh,
                                   resolution=args.resolution,
                                   save_voxel_path=out_path.parent / "voxel.npz",
                                   try_cpu=args.try_cpu)
            logger.debug(f"Ran Kaolin pipeline in {time() - restart:.2f}s.")
        elif args.mode == "fuse":
            restart = time()
            rotations = get_views(get_points(args.n_views))
            logger.debug(f"Generated rotations in {time() - restart:.2f}s.")

            restart = time()
            depthmaps = render(mesh,
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
                               not args.no_erosion,
                               args.flip_faces,
                               show=False)
            logger.debug(f"Rendered depth maps in {time() - restart:.2f}s.")

            restart = time()
            tsdf = fuse(depthmaps,
                        rotations,
                        args.resolution,
                        args.fx,
                        args.fy,
                        args.cx,
                        args.cy)
            logger.debug(f"Fused depth maps in {time() - restart:.2f}s.")

            restart = time()
            mesh = extract(grid=-tsdf,
                           level=0,
                           resolution=args.resolution,
                           return_type="trimesh" if args.use_trimesh else "dict")
        elif args.mode == "carve":
            raise NotImplementedError("Voxel carving is not implemented yet.")

        vertices, faces = get_vertices_and_faces(mesh)
        if len(vertices) == 0 or len(faces) == 0:
            logger.warning(f"Extracted mesh is empty. Skipping.")
            return
        logger.debug(f"Extracted mesh ({len(vertices)} vertices, {len(faces)} faces) in {time() - restart:.2f}s.")

        if args.script_dir is not None:
            restart = time()
            script_dir = args.script_dir.expanduser().resolve()
            assert script_dir.is_dir(), f"Script dir {script_dir} is not a directory."
            scripts = load_scripts(script_dir, num_vertices=len(vertices))
            mesh = process(mesh, scripts)

            vertices, faces = get_vertices_and_faces(mesh)
            if len(vertices) == 0 or len(faces) == 0:
                logger.warning(f"Filtered mesh is empty. Skipping.")
                return
            logger.debug(f"Filtered mesh ({len(vertices)} vertices, {len(faces)} faces) in {time() - restart:.2f}s.")

        if not args.no_normalization:
            restart = time()
            mesh, _, _ = normalize(mesh, translation=-translation * 1 / scale, scale=1 / scale)
            logger.debug(f"Normalized mesh in {time() - restart:.2f}s.")

        if args.check_watertight:
            restart = time()
            vertices, faces = get_vertices_and_faces(mesh)
            _mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not _mesh.is_watertight:
                logger.warning(f"Mesh {in_path} is not watertight. Skipping.")
                return
            logger.debug(f"Checked watertightness in {time() - restart:.2f}s.")

        restart = time()
        save(mesh, out_path, args.precision)
        logger.debug(f"Saved mesh in {time() - restart:.2f}s.")

        del mesh
        gc.collect()
    except Exception as e:
        logger.exception(e)
    logger.debug(f"Runtime: {time() - start:.2f}s.\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("in_dir", type=Path, help="Path to input directory.")
    parser.add_argument("--out_dir", type=Path, help="Path to output directory.")
    parser.add_argument("--in_format", type=str, default=".obj", help="Input file format.")
    parser.add_argument("--out_format", type=str, default=".off", choices=[".obj", ".off", ".ply", ".stl"],
                        help="Output file format.")
    parser.add_argument("--recursion_depth", type=int, default=-1, help="Depth of recursive glob pattern matching.")
    parser.add_argument("--script_dir", type=Path, default="./meshlab_filter_scripts",
                        help="Path to directory containing MeshLab scripts.")
    parser.add_argument("--width", type=int, default=640, help="Width of the depth map.")
    parser.add_argument("--height", type=int, default=640, help="Height of the depth map.")
    parser.add_argument("--fx", type=float, default=640, help="Focal length in x.")
    parser.add_argument("--fy", type=float, default=640, help="Focal length in y.")
    parser.add_argument("--cx", type=float, default=320, help="Principal point in x.")
    parser.add_argument("--cy", type=float, default=320, help="Principal point in y.")
    parser.add_argument("--znear", type=float, default=0.25, help="Near clipping plane.")
    parser.add_argument("--zfar", type=float, default=1.75, help="Far clipping plane.")
    parser.add_argument("--padding", type=float, default=0.1, help="Relative padding applied on each side.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution of the TSDF fusion voxel grid.")
    parser.add_argument("--depth_offset", type=float, default=1.5,
                        help="Thicken object through offsetting of rendered depth maps.")
    parser.add_argument("--no_erosion", action="store_true",
                        help="Do not erode rendered depth maps to thicken thin structures.")
    parser.add_argument("--no_normalization", action="store_true", help="Do not normalize the mesh.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs.")
    parser.add_argument("--n_views", type=int, default=100, help="Number of views to render.")
    parser.add_argument("--precision", type=int, default=16, choices=[16, 32, 64], help="Data precision.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--flip_faces", action="store_true", help="Flip faces (i.e. invert normals) of the mesh.")
    parser.add_argument("--use_trimesh", action="store_true", help="Use trimesh for loading and saving meshes.")
    parser.add_argument("--mode", type=str, default="fuse", choices=["fuse", "carve", "fill", "script"],
                        help="Apply TSDF fusion, voxel carving or hole filling to the meshes. Use 'script' to only"
                             "apply MeshLab filter scripts from the provided 'script_dir'.")
    parser.add_argument("--sort", action="store_true", help="Sort files before processing.")
    parser.add_argument("--check", action="store_true", help="Check results.")
    parser.add_argument("--check_watertight", action="store_true", help="Verify that generated mesh is watertight.")
    parser.add_argument("--fix", action="store_true", help="Fix results that failed check.")
    parser.add_argument("--try_cpu", action="store_true", help="Fallback to CPU if GPU fails.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    if args.use_trimesh:
        logging.getLogger("trimesh").setLevel(logging.ERROR)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    in_dir = args.in_dir.expanduser().resolve()
    logger.debug(f"Globbing paths from {in_dir}.")
    if args.recursion_depth == -1:
        files = list(in_dir.rglob(f"*{args.in_format}"))
    else:
        pattern = f"{'/'.join(['*' for _ in range(args.recursion_depth)])}/*{args.in_format}"
        files = list(in_dir.glob(pattern))
    shuffle(files)
    if args.sort:
        files = sorted(files)
    logger.debug(f"Found {len(files)} files.")

    # tracemalloc.start()

    progress = tqdm(desc="Mesh Fusion", total=len(files), disable=args.verbose)
    with tqdm_joblib(progress):
        Parallel(n_jobs=args.n_jobs)(delayed(run)(file, args) for file in files)


if __name__ == "__main__":
    main()
