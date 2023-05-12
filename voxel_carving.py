import open3d as o3d
import numpy as np
import mcubes


def xyz_spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    r = np.sqrt(x * x + y * y + z * z)
    r_x = np.arccos(y / r)
    r_y = np.arctan2(z, x)
    return [r, r_x, r_y]


def get_rotation_matrix(r_x, r_y):
    rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                        [0, np.sin(r_x), np.cos(r_x)]])
    rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], [0, 1, 0],
                        [-np.sin(r_y), 0, np.cos(r_y)]])
    return rot_y.dot(rot_x)


def get_extrinsic(xyz):
    rvec = xyz_spherical(xyz)
    r = get_rotation_matrix(rvec[1], rvec[2])
    t = np.asarray([0, 0, 2]).transpose()
    trans = np.eye(4)
    trans[:3, :3] = r
    trans[:3, 3] = t
    return trans


def preprocess(model, return_center_scale=False):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 2.0
    vertices = np.asarray(model.vertices)
    vertices -= center
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    if return_center_scale:
        return model, center, scale
    return model


def voxel_carving(mesh, cubic_size, voxel_resolution, w=300, h=300):
    mesh.compute_vertex_normals()
    camera_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=1.0, resolution=10)

    # Setup dense voxel grid.
    vc = o3d.geometry.VoxelGrid().create_dense(
        width=cubic_size,
        height=cubic_size,
        depth=cubic_size,
        voxel_size=cubic_size / voxel_resolution,
        origin=[-cubic_size / 2.0, -cubic_size / 2.0, -cubic_size / 2.0],
        color=[1.0, 0.7, 0.0])

    # Rescale geometry.
    camera_sphere = preprocess(camera_sphere)
    mesh = preprocess(mesh)

    # Setup visualizer to render depthmaps.
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=w, height=h, visible=False)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    param = ctr.convert_to_pinhole_camera_parameters()

    # Carve voxel grid.
    centers_pts = np.zeros((len(camera_sphere.vertices), 3))
    for cid, xyz in enumerate(camera_sphere.vertices):
        # Get new camera pose.
        trans = get_extrinsic(xyz)
        param.extrinsic = trans
        c = np.linalg.inv(trans).dot(np.asarray([0, 0, 0, 1]).transpose())
        centers_pts[cid, :] = c[:3]
        ctr.convert_from_pinhole_camera_parameters(param)

        # Capture depth image and make a point cloud.
        vis.poll_events()
        vis.update_renderer()
        depth = vis.capture_depth_float_buffer(False)

        # Depth map carving method.
        # vc.carve_depth_map(o3d.geometry.Image(depth), param)
        vc.carve_silhouette(o3d.geometry.Image(depth), param)
        print("Carve view %03d/%03d" % (cid + 1, len(camera_sphere.vertices)))
    vis.destroy_window()

    return vc


if __name__ == "__main__":
    cubic_size = 2.0
    voxel_resolution = 64.0
    num_points = int(voxel_resolution)

    mesh_hash = '1a3b35be7a0acb2d9f2366ce3e663402'
    mesh_hash = '1a4ef4a2a639f172f13d1237e1429e9e'
    mesh_path = '/run/media/matthias/2C20BCA320BC7604/datasets/shapenet/ShapeNetCore.v1/03797390/1a97f3c83016abca21d0de04f408950f/model.obj'
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh, center, scale = preprocess(mesh, return_center_scale=True)

    vertices = np.asarray(mesh.vertices)
    min_val = vertices.min(axis=0)
    max_val = vertices.max(axis=0)
    print((min_val + max_val) / 2, (max_val - min_val).max())

    carved_voxels = voxel_carving(mesh, cubic_size, voxel_resolution, w=300, h=300)
    print("Carved voxels ...")
    print(carved_voxels)

    voxel_indices = np.asarray([voxel.grid_index for voxel in carved_voxels.get_voxels()])

    points = voxel_indices / (voxel_resolution - 1)
    points = (points - 0.5) * cubic_size

    min_val = points.min(axis=0)
    max_val = points.max(axis=0)
    print((min_val + max_val) / 2, (max_val - min_val).max())

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.paint_uniform_color([0.7, 0.1, 0.1])

    occupancies = np.zeros((num_points,) * 3, dtype=bool)
    occupancies[tuple(voxel_indices.T)] = True
    # occ_hat_padded = np.pad(occupancies.astype(float), 1, "constant", constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(occupancies, 0.5)
    # vertices -= 1
    vertices /= (voxel_resolution - 1)
    vertices = (vertices - 0.5) * cubic_size

    min_val = vertices.min(axis=0)
    max_val = vertices.max(axis=0)
    print((min_val + max_val) / 2, (max_val - min_val).max())

    carved_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    carved_mesh.compute_vertex_normals()
    carved_mesh.paint_uniform_color([0.1, 0.1, 0.7])

    o3d.visualization.draw_geometries([carved_mesh, pcd], mesh_show_back_face=True)
