import glob
import os

import cv2
import numpy as np

import config
from step5_create_mask import write_png
from utils.reader import Labels


def clean_obstacles(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    good_components = []
    for x in range(1, n):
        stat = stats[x]
        is_good = \
            stat[0] <= 0 or \
            stat[1] <= 0 or \
            stat[0] + stat[2] >= mask.shape[1]

        is_good = is_good or stat[4] >= config.min_obstacle_pixel_count
        if not is_good:
            good_components.append(x)

    for component in good_components:
        mask[labels == component] = 0


def clean_ground(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    biggest_component = None
    biggest_pixels = None
    for x in range(1, n):
        pixels = stats[x][cv2.CC_STAT_AREA]
        if biggest_pixels is None or pixels > biggest_pixels:
            biggest_pixels = pixels
            biggest_component = x

    for x in range(1, n):
        if x != biggest_component:
            mask[np.where(labels == x)] = 0


def clean_cc(name):
    print("cleaning", name)
    os.makedirs(os.path.join(name, "label_clean"), exist_ok=True)
    labels = Labels(name)
    for x in range(labels.count()):
        label = labels.label(x)

        obstacle = (label[..., 2] == 255).astype(np.uint8) * 255
        ground = (label[..., 1] == 255).astype(np.uint8) * 255

        clean_ground(ground)
        clean_obstacles(obstacle)

        mask = np.ones(shape=(label.shape[:2]), dtype=np.uint8) * 255
        mask[ground == 255] = 0
        mask[obstacle == 255] = 1

        filename = os.path.basename(name) + "_%05d.png" % x
        write_png(mask, name + "/label_clean/" + filename)

        # DEBUG: show mask before cleaning
        if config.debug_step6:
            cv2.imshow("label", cv2.addWeighted(labels.label(x, cleaned=False), (1. - 0.5), labels.color(x), 0.5, 0))
            cv2.waitKey(0)

        # DEBUG: show final mask
        if config.debug_step6:
            cv2.imshow("label", cv2.addWeighted(labels.label(x, cleaned=True), (1. - 0.5), labels.color(x), 0.5, 0))
            cv2.waitKey(0)


def clean_all_cc():
    folders = glob.glob("/data/**/label")
    for folder in folders:
        dir = os.path.dirname(folder)
        if os.path.exists(dir + "/label_clean"):
            continue
        clean_cc(dir)


if __name__ == "__main__":
    clean_all_cc()
