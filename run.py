import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

from typing import Any, List, Tuple
from pathlib import Path
import contextlib
from argparse import ArgumentParser
from joblib import Parallel, delayed
import math
import time

from tqdm import tqdm
import trimesh
from trimesh import Trimesh
import numpy as np
import mcubes
import pyrender
import pymeshlab
from scipy import ndimage
from PIL import Image

import librender
import libfusiongpu as libfusion


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

        R_x = np.array([[1, 0, 0], [0, math.cos(latitude), -math.sin(latitude)], [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)], [0, 1, 0], [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)
        Rs.append(R)

    return Rs

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
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


def convert(path: Path, args: Any):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(path))
    ms.save_current_mesh(file_name=str(path).replace(args.in_format, args.out_format),
                         save_vertex_color=False,
                         save_vertex_coord=False,
                         save_face_color=False,
                         save_polygonal=False)



def scale(path: Path, args: Any) -> Trimesh:
    mesh = trimesh.load(str(path).replace(args.in_format, args.out_format), process=False)
    mesh.apply_scale(1.0 / mesh.scale)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(1.0 + args.padding)
    return mesh


def render(mesh: Trimesh, args: any) -> List[np.ndarray]:
    renderer = pyrender.OffscreenRenderer(args.width, args.height)
    camera = pyrender.IntrinsicsCamera(args.fx, args.fy, args.cx, args.cy, znear=args.znear, zfar=args.zfar)

    depth_maps = list()
    for R in get_views(get_points(args.n_views)):
        """
        vertices = R.dot(mesh.vertices.astype(np.float64).T)
        vertices[2, :] += 1
        faces = mesh.faces.astype(np.float64)
        """

        trafo = np.eye(4)
        trafo[:3, 0] = R[:3, 0]
        trafo[:3, 1] = -R[:3, 1]
        trafo[:3, 2] = R[:3, 2]
        trafo[:3, 3] = np.array([0, 0, -1])
        
        mesh_t = mesh.copy()
        mesh_t.apply_transform(trafo)

        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh_t))
        scene.add(camera)

        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        depth[depth == 0] = args.zfar 

        """
        render_intrinsics = np.array([
            args.fx,
            args.fy,
            args.cx,
            args.cy
        ], dtype=float)
        znf = np.array([args.znear, args.zfar], dtype=float)
        image_size = np.array([args.height, args.width], dtype=np.int32)

        faces += 1
        depth, mask, img = librender.render(vertices.copy(), faces.T.copy(), render_intrinsics, znf, image_size)
        """

        # print(depth.shape, depth.min(), depth.max(), depth.mean())
        # Image.fromarray(((depth / depth.max()) * 255).astype(np.uint8)).convert('L').save('depth_new.png')
        # time.sleep(100)
        
        depth -= 1.5 * (1 / args.resolution)
        depth = ndimage.grey_erosion(depth, size=(3, 3))
        depth_maps.append(depth)

    renderer.delete()
    return depth_maps


def fuse(depth_maps: List[np.ndarray], args: Any) -> np.ndarray:
    Ks = np.array([[args.fx, 0, args.cx],
                   [0, args.fy, args.cy],
                   [0, 0, 1]]).reshape((1, 3, 3))

    Ks = np.repeat(Ks, len(depth_maps), axis=0).astype(np.float32)
    Rs = np.array(get_views(get_points(args.n_views))).astype(np.float32)
    Ts = np.array([np.array([0, 0, 1]) for _ in range(len(Rs))]).astype(np.float32)
    depth_maps = np.array(depth_maps).astype(np.float32)

    views = libfusion.PyViews(depth_maps, Ks, Rs, Ts)
    truncation_factor = 10 * (1 / args.resolution)
    tsdf = libfusion.tsdf_gpu(views, args.resolution, args.resolution, args.resolution, 1 / args.resolution, truncation_factor, False)
    tsdf = np.transpose(tsdf[0], [2, 1, 0])

    return tsdf


def extract(path: Path, tsdf: np.ndarray, args: Any) -> np.ndarray:
    tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)

    vertices, triangles = mcubes.marching_cubes(-tsdf, 0)

    vertices -= 1
    vertices /= args.resolution
    vertices -= 0.5

    print(Trimesh(vertices=vertices, faces=triangles))
    print(str(path).replace(args.in_format, args.out_format))
    mcubes.export_off(vertices, triangles, str(path).replace(args.in_format, args.out_format))


def clean():
    pass


def run(path: Path, args: Any):
    convert(path, args)
    mesh = scale(path, args)
    depth_maps = render(mesh, args)
    tsdf = fuse(depth_maps, args)
    extract(path, tsdf, args)
    clean()


def main():
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str, help="Path to input directory.")
    parser.add_argument("--out_dir", type=str, help="Path to output directory.")
    parser.add_argument("--in_format", type=str, default="obj", help="Input file format.")
    parser.add_argument("--out_format", type=str, default="off", help="Output file format.")
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
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs.")
    parser.add_argument("--n_views", type=int, default=100, help="Number of views to render.")
    args = parser.parse_args()

    files = sorted(Path(args.in_dir).rglob(f"*.{args.in_format}"))
    with tqdm_joblib(tqdm(desc="Mesh Fusion", total=len(files))):
        Parallel(n_jobs=args.n_jobs)(delayed(run)(file, args) for file in files)


if __name__ == "__main__":
    main()

