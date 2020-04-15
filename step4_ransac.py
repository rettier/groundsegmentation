import gc
import glob
import os
import random
import sys

import numpy as np

import pcl

import config
from utils.reader import Reader


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    return np.linalg.svd(xyzs)[-1][-1, :]


def run_ransac(data, sample_size, goal_inliers, max_iterations,
               stop_at_goal=True, random_seed=None, thresh=0.05):
    data = augment(data)

    best_ic = 0
    best_model = None
    if random_seed:
        random.seed(random_seed)

    idx = list(range(data.shape[0]))
    for i in range(max_iterations):
        s = data[random.sample(idx, sample_size)]
        m = estimate(s)
        ic = np.count_nonzero(np.abs(m.dot(data.T)) < thresh)
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if stop_at_goal and ic > goal_inliers:
                break

    return best_model, best_ic


def calculate_ransac(name):
    print("calculating ransac planes for", name)
    reader = Reader(name, color=False, infra=False, coords=True)

    good_idx = np.zeros(shape=(reader.coords.shape[0], reader.coords.shape[1]), dtype=np.bool)
    planes = np.ndarray(shape=(reader.coords.shape[0], 4), dtype=np.float32)

    for i in range(reader.coords.shape[0]):
        coords = reader.coords_not_nan(i)
        p = pcl.PointCloud(coords)
        seg = config.make_segmenter(p)
        indices, model = seg.segment()

        del p
        gc.collect()
        sys.stdout.write(".")
        sys.stdout.flush()

        good_idx[i, indices] = True
        planes[i, :] = model

        # DEBUG: show segmented cloud
        if config.debug_step4:
            from utils.pcl import draw_pointcloud, color_inlier
            draw_pointcloud(coords, colors=color_inlier(coords, indices), plane=model)

    print(" done.")
    np.save(reader.path("good_idx", ext="npy"), good_idx, allow_pickle=False)
    np.save(reader.path("planes", ext="npy"), planes, allow_pickle=False)


def calculate_all_ransac():
    folders = glob.glob(os.path.join(config.data_directory, "**/coords.npy"))
    for folder in folders:
        dir = os.path.dirname(folder)
        if os.path.exists(dir + "/planes.npy"):
            continue

        calculate_ransac(dir)


if __name__ == "__main__":
    calculate_all_ransac()
