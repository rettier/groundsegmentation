import numpy as np
import open3d as o3d


def color_inlier(coords, inlier):
    colors = np.ndarray(shape=(coords.shape[0], 3), dtype=np.float32)
    colors[:] = [1, 0, 0]
    colors[inlier] = [0, 1, 0]
    return colors


def draw_pointcloud(coords, colors=None, plane=None, axis=True):
    draw = []
    if axis:
        draw.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0]))
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(coords)
    if colors is not None:
        pcl.colors = o3d.utility.Vector3dVector(colors)
    draw.append(pcl)

    if plane is not None:
        a, b, c, d = plane
        size = 1.0
        xx = np.array([-size, -size, size, size])
        yy = np.array([-size, 0, 0, -size])
        zz = (-d - a * xx - b * yy) / c

        points = np.stack   ([xx, yy, zz]).T
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        colors = [[0, 0, 0] for x in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        draw.append(line_set)

    o3d.visualization.draw_geometries(draw)
