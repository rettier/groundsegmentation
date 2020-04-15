from __future__ import print_function, unicode_literals

import glob
import os
import pickle

import numpy as np
import rosbag
from cv_bridge import CvBridge

import config


class Extract:
    extrinsics_topics = [
        "/camera/extrinsics/depth_to_color",
        "/camera/extrinsics/depth_to_infra1",
        "/camera/extrinsics/depth_to_infra2"
    ]

    camera_info_topics = [
        "/camera/color/camera_info",
        "/camera/depth/camera_info",
        "/camera/infra1/camera_info",
        "/camera/infra2/camera_info"
    ]

    def __init__(self, bag, to):
        self.path = to
        self.bag = rosbag.Bag(bag)
        try:
            os.makedirs(self.path)
        except:
            pass

    def commonpath(self, p, ext=""):
        common_dir = os.path.join(self.path, "../common")
        try:
          os.makedirs(common_dir)
        except:
          pass
        return os.path.join(common_dir, p) + (("." + ext) if ext else "")

    def outpath(self, p, ext=""):
        return os.path.join(self.path, p) + (("." + ext) if ext else "")

    def extract_extrinsics(self):
        done = set()
        for topic, msg, t in self.bag.read_messages(topics=self.extrinsics_topics):
            extrinsic = topic.split("/")[-1]
            if len(done) == len(self.extrinsics_topics):
                break
            if extrinsic in done:
                continue
            done.add(extrinsic)
            rotation = msg.rotation
            translation = msg.translation
            with open(self.commonpath("extrinsic_" + extrinsic, ext="pkl"), "w") as f:
                pickle.dump({"name": extrinsic, "rotation": rotation, "translation": translation}, f)

    def extract_camera_info(self):
        done = set()
        for topic, msg, t in self.bag.read_messages(topics=self.camera_info_topics):
            camera = topic.split("/")[-2]
            if len(done) == len(self.camera_info_topics):
                break
            if camera in done:
                continue
            done.add(camera)
            attrs = "height,width,distortion_model,D,K,R,P,binning_x,binning_y".split(",")
            obj = {x: getattr(msg, x) for x in attrs}
            with open(self.commonpath("camera_" + camera, ext="pkl"), "w") as f:
                pickle.dump(obj, f)

    def extract_sync_images(self):
        bridge = CvBridge()
        topics = [config.topic_infra1, config.topic_infra2, config.topic_color]
        frames = {t: {} for t in topics}
        for topic, msg, t in self.bag.read_messages(topics):
            time = msg.header.stamp.secs + float(msg.header.stamp.nsecs) / 1.e9
            frames[topic][time] = bridge.imgmsg_to_cv2(msg)

        times = {t: np.array(list(frames[t].keys()), dtype=np.float64) for t in topics}

        matched_ir1 = []
        matched_ir2 = []
        matched_color = []

        for i, t in enumerate(sorted(times[config.topic_infra1])):
            if t in frames[config.topic_infra2] and t in frames[config.topic_color]:
                matched_ir1.append(frames[config.topic_infra1][t])
                matched_ir2.append(frames[config.topic_infra2][t])
                matched_color.append(frames[config.topic_color][t])

        print("found {} matching paris of {} total frames".format(len(matched_color), len(times[config.topic_infra1])))
        np_images = np.array(matched_ir1)
        np_images.dump(self.outpath("infra1", ext="npy"))
        np_images = np.array(matched_ir2)
        np_images.dump(self.outpath("infra2", ext="npy"))
        np_images = np.array(matched_color)
        np_images.dump(self.outpath("color", ext="npy"))

    def extract(self):
        self.extract_extrinsics()
        self.extract_camera_info()
        self.extract_sync_images()


def bag_name_to_filename(x):
    return x.replace(folder, "").replace("/", "_").replace(".bag", "")


if __name__ == "__main__":
    folder = config.bagfile_folder
    print("looking for bagfiles in", folder)

    for x in glob.glob(os.path.join(config.bagfile_folder, "**/*.bag")):
        dst_folder = bag_name_to_filename(x)
        dst_path = os.path.join(config.data_directory, dst_folder)
        if os.path.exists(dst_path):
            continue

        print("extracting")
        print("from ", x, "to", )
        e = Extract(
            bag=x,
            to=dst_path
        )
        e.extract()
