import gc
import glob
import os
from time import sleep

import cv2
import numpy as np

import config
from utils.reader import Reader


def calculate_pointclouds(name):
    print("calculating pointclouds for {}".format(name))
    reader = Reader(name, color=False, infra=True)

    shape = reader.infra1.shape
    all_coords = np.ndarray(shape=(shape[0], shape[1] * shape[2], 3), dtype=np.float32)
    for i in range(reader.infra1.shape[0]):
        ir1 = reader.infra1[i, ...]
        ir2 = reader.infra2[i, ...]

        stereo = config.get_stereo()
        disp = stereo.compute(ir1, ir2).astype(np.float32) * (1 / 16.0)

        K = reader.camera_infra1["K"]
        cx, cy, f = K[0, 2], K[1, 2], K[0, 0]
        Tx = -reader.depth_to_infra2["translation"][0]
        Q = np.array((
            (1, 0, 0, -cx),
            (0, 1, 0, -cy),
            (0, 0, 0, f),
            (0, 0, 1 / Tx, 0)
        ))
        coords = cv2.reprojectImageTo3D(disp, Q, ddepth=cv2.CV_32FC3).reshape(-1, 3)
        coords[np.any(coords == np.inf, axis=1), :] = np.nan
        coords[coords[:, 2] < 0, :] = np.nan
        coords[coords[:, 2] > 10, :] = np.nan

        # DEBUG: show pointcloud
        if config.debug_step3:
            from utils.pcl import draw_pointcloud
            draw_pointcloud(coords)

        all_coords[i, ...] = coords

    path = reader.path("coords", ext="npy")
    del reader
    gc.collect()
    sleep(1)

    np.save(path, all_coords, allow_pickle=False)


def calculate_all_pointclouds():
    folders = glob.glob(os.path.join(config.data_directory, "**/color.npy"))
    for folder in folders:
        dir = os.path.dirname(folder)
        if os.path.exists(dir + "/coords.npy"):
            continue

        calculate_pointclouds(dir)


if __name__ == "__main__":
    calculate_all_pointclouds()
