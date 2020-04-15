import os
import re
import shutil
from collections import defaultdict

import cv2
import numpy as np

import config
from utils.reader import Reader


def gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


_last_frame_gauss_cache = None


def get_change(last_frame, new_frame):
    global _last_frame_gauss_cache
    if _last_frame_gauss_cache is None or _last_frame_gauss_cache[0] is not last_frame:
        _last_frame_gauss_cache = [last_frame, cv2.blur(cv2.resize(last_frame, (320, 240)), (3, 3))]
    new_frame_gauss = cv2.blur(cv2.resize(new_frame, (320, 240)), (3, 3))
    return np.sum(cv2.absdiff(new_frame_gauss, _last_frame_gauss_cache[1])) / (320 * 240)


def get_sharpness(new_frame):
    new_frame = cv2.resize(new_frame, (320, 240))
    laplacian = cv2.Laplacian(new_frame, cv2.CV_64F)
    mean, std_dev = cv2.meanStdDev(laplacian)
    return std_dev[0][0] * std_dev[0][0]


def get_sharpest_frame(frames, idx, batch_size):
    indices, sharpness = [], []
    for x in range(batch_size):
        indices.append(idx)
        sharpness.append(get_sharpness(gray(frames[idx])))
        idx += 1
        if idx >= frames.shape[0]:
            break
    return idx, indices[np.argmax(sharpness)]


def reduce_folder(folder):
    reader = Reader(folder, color=True, infra=True)

    # -------------------------
    # parameters
    change_threshold = config.change_threshold
    batch_size = config.sharpest_frame_batch_size
    # -------------------------

    good_idx = [0]
    last_frame = gray(reader.color[0, ...])
    idx = 1
    while idx < reader.color.shape[0]:
        frame = gray(reader.color[idx, ...])
        change = get_change(last_frame, frame)
        if change > change_threshold:
            last_frame = frame
            idx, sharp_idx = get_sharpest_frame(reader.color, idx, batch_size)
            good_idx.append(sharp_idx)
        else:
            idx += 1

    # DEBUG: view all selected image
    if config.debug_step2:
        for idx in good_idx:
            cv2.imshow("selected", cv2.cvtColor(reader.color[idx, ...], cv2.COLOR_BGR2RGB))
            cv2.waitKey(0)

    good_color = reader.color[good_idx, ...]
    good_infra1 = reader.infra1[good_idx, ...]
    good_infra2 = reader.infra2[good_idx, ...]

    print("reduced {} from {} frames to {}".format(folder, reader.color.shape[0], len(good_idx)))

    good_infra1.dump(reader.path("infra1_reduced", ext="npy"))
    good_infra2.dump(reader.path("infra2_reduced", ext="npy"))
    good_color.dump(reader.path("color_reduced", ext="npy"))


def combine_recordings(reduced, delete_old=True):
    collections = defaultdict(list)
    for name in reduced:
        collections[name.split("_")[0]].append(name)

    for name, children in collections.items():
        print("reducing", name)

        all_color = np.ndarray(shape=(0, config.color_height, config.color_width, 3), dtype=np.uint8)
        all_infra1 = np.ndarray(shape=(0, config.ir_height, config.ir_width), dtype=np.uint8)
        all_infra2 = np.copy(all_infra1)

        for child in children:
            reader = Reader(child, suffix="_reduced")
            all_color = np.concatenate([all_color, reader.color], axis=0)
            all_infra1 = np.concatenate([all_infra1, reader.infra1], axis=0)
            all_infra2 = np.concatenate([all_infra2, reader.infra2], axis=0)

        os.makedirs(name, exist_ok=True)
        reader = Reader(name, infra=False, color=False)
        all_infra1.dump(reader.path("infra1", ext="npy"))
        all_infra2.dump(reader.path("infra2", ext="npy"))
        all_color.dump(reader.path("color", ext="npy"))

        if delete_old:
            for child in children:
                print("deleting", child)
                shutil.rmtree(child)


def reduce_data():
    regex = re.compile("([\w\d]+)_.*_[\d]+")

    reduced = []
    folders = os.listdir(config.data_directory)
    for folder in folders:
        if regex.match(folder):
            abs_folder = os.path.join(config.data_directory, folder)
            reduce_folder(abs_folder)
            reduced.append(abs_folder)

    combine_recordings(reduced)


if __name__ == "__main__":
    reduce_data()
