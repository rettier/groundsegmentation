from __future__ import print_function, unicode_literals

import glob
import os
import pickle

import cv2
import numpy as np


class Reader:
    camera_color = camera_infra1 = camera_infra2 = camera_depth = dict()
    depth_to_color = depth_to_infra1 = depth_to_infra2 = dict()

    def path(self, *p, **kwargs):
        ext = kwargs.get("ext")
        return os.path.realpath(os.path.join(self.folder, *p) + (("." + ext) if ext else ""))

    def load_camera_info(self):
        for x in ["depth", "color", "infra1", "infra2"]:
            with open(self.path("..", "common", "camera_" + x, ext="pkl"), "rb") as f:
                data = pickle.load(f)
            data["K_1d"] = data["K"]
            data["K"] = np.array(data["K"]).reshape(3, 3)
            setattr(self, "camera_" + x, data)

    def load_extrinsics(self):
        for x in ["depth_to_color", "depth_to_infra1", "depth_to_infra2"]:
            with open(self.path("..", "common", "extrinsic_{}".format(x), ext="pkl"), "rb") as f:
                data = pickle.load(f)
            data["rotation"] = np.array(data["rotation"])
            data["translation"] = np.array(data["translation"])
            setattr(self, x, data)

    def coords_not_nan(self, i):
        coords = self.coords[i, ...]
        nan_map = (np.logical_not(np.any(np.isnan(coords), axis=1)))
        return coords[nan_map, :]

    def model(self, i):
        return self.planes[i, ...]

    def inlier(self, i):
        return np.where(self.good_idx[i, :])

    def __init__(self, folder, infra=True, color=True, coords=False, planes=False, suffix=""):
        self.folder = folder
        self.load_camera_info()
        self.load_extrinsics()
        self.suffix = suffix
        self.is_inverse = self.folder.rstrip("/").endswith("_inv")
        self.below_is_obstacle = os.path.exists(self.folder + "/below_nok")

        if infra:
            self.infra1 = np.load(self.path("infra1" + suffix, ext="npy"), allow_pickle=True, encoding="bytes")
            self.infra2 = np.load(self.path("infra2" + suffix, ext="npy"), allow_pickle=True, encoding="bytes")

        if color:
            self.color = np.load(self.path("color" + suffix, ext="npy"), allow_pickle=True, encoding="bytes")

        if coords:
            self.coords = np.load(self.path("coords" + suffix, ext="npy"), allow_pickle=True, encoding="bytes")

        if planes:
            self.planes = np.load(self.path("planes" + suffix, ext="npy"), allow_pickle=True, encoding="bytes")
            self.good_idx = np.load(self.path("good_idx" + suffix, ext="npy"), allow_pickle=True, encoding="bytes")


class Labels:
    def __init__(self, folder):
        self.folder = folder

    def path(self, *p, **kwargs):
        ext = kwargs.get("ext")
        return os.path.realpath(os.path.join(self.folder, *p) + (("." + ext) if ext else ""))

    def count(self):
        return len(glob.glob(self.path("color/*.png")))

    def names(self):
        for x in glob.glob(self.path("color/*.png")):
            yield x.split("/")[-1]

    def label(self, i=0, cleaned=False, name=None):

        if cleaned:
            folder = "label_clean/"
        else:
            folder = "label/"
        if name:
            return cv2.imread(self.path(folder, name))
        else:
            return cv2.imread(self.path(folder, os.path.basename(self.folder) + "_%05d.png" % i))

    def load_good_bad_index(self):
        try:
            with open(self.path("good.txt")) as f:
                good = {x for x in f.read().splitlines(keepends=False) if x}
        except FileNotFoundError:
            print(self.path("good.txt"))
            good = set()

        try:
            with open(self.path("bad.txt")) as f:
                bad = {x for x in f.read().splitlines(keepends=False) if x}
        except FileNotFoundError:
            bad = set()

        return good, bad

    def save_good_bad_index(self, good, bad):
        with open(self.path("good.txt"), "w") as f:
            f.writelines([x+"\n" for x in good])
        with open(self.path("bad.txt"), "w") as f:
            f.writelines([x+"\n" for x in bad])

    def color(self, i=0, name=None):
        if name:
            return cv2.imread(self.path("color", name))
        else:
            return cv2.imread(self.path("color", os.path.basename(self.folder) + "_%05d.png" % i))
