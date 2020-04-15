import glob
import os

import cv2
import numpy as np
import png
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

import config
from utils.reader import Reader, Labels


def points_to_image(coords, reader, value=None):
    if not len(coords):
        tempimg = np.zeros(shape=(config.color_height, config.color_width), dtype=np.float32)
        return tempimg[..., np.newaxis]

    imgpts, jac = cv2.projectPoints(coords,
                                    rvec=reader.depth_to_color["rotation"].reshape(3, 3).transpose(),
                                    tvec=reader.depth_to_color["translation"],
                                    cameraMatrix=reader.camera_color["K"],
                                    distCoeffs=None)

    imgpts = np.int32(imgpts).reshape(-1, 2)
    imgpts[imgpts[:, 1] >= config.color_height] = 0
    imgpts[imgpts[:, 0] >= config.color_width] = 0
    imgpts[imgpts < 0] = 0

    tempimg = np.zeros(shape=(config.color_height, config.color_width), dtype=np.float32)
    tempimg[imgpts[:, 1], imgpts[:, 0]] = value if value is not None else 1
    kernel = np.ones((3, 3), np.float32)
    tempimg = cv2.morphologyEx(tempimg, cv2.MORPH_DILATE, kernel)
    return tempimg[..., np.newaxis]


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4), dtype=np.float32)
    axyz[:, :3] = xyzs
    return axyz


def dist_from_plane(coeffs, xyz):
    return coeffs.dot(xyz.T)


def write_png(mask, path):
    fullpalette = [(0, 0, 0), ] * 255
    fullpalette[0] = (0, 255, 0)
    fullpalette[1] = (255, 0, 0)
    out = png.Writer(mask.shape[1], mask.shape[0], palette=fullpalette)
    with open(path, "wb") as f:
        out.write(f, mask)


def calculate_masks(name):
    print("calculating masks", name)

    os.makedirs(os.path.join(name, "label"), exist_ok=True)
    os.makedirs(os.path.join(name, "color"), exist_ok=True)

    # -------------------
    # parameter
    below_ok = 0.01
    abote_nok = 0.05
    # ------------------

    reader = Reader(name, color=True, infra=False, coords=True, planes=True)
    for i in range(reader.coords.shape[0]):
        color = cv2.cvtColor(reader.color[i, ...], cv2.COLOR_BGR2RGB)
        coords = reader.coords_not_nan(i)
        coords_4 = augment(coords)

        plane = reader.model(i)
        idx_inlier = reader.inlier(i)
        dZ = dist_from_plane(plane, coords_4)
        if reader.is_inverse:
            dZ *= -1

        mask_inlier = np.zeros(len(coords), dtype=bool)
        mask_obstacle = np.zeros(len(coords), dtype=bool)
        mask_ignore = np.zeros(len(coords), dtype=bool)

        # inliers from ransac
        mask_inlier[idx_inlier] = True

        # everything too far above ground is not an inlier, even if ransac saied so
        mask_inlier[dZ > abote_nok] = False

        # everything a little below ground is an inlier, even if ransac didnt agree
        if reader.below_is_obstacle:
            mask_inlier[np.logical_and(dZ <= 0, dZ > -below_ok)] = True
        else:
            mask_inlier[dZ <= 0] = True

            # everything 0.7 meter below the ground will be ignored (mostly errors)
            mask_ignore[dZ < -0.7] = True

        # everything 10 meters away will be ignored
        mask_ignore[coords[:, 2] > 5] = True

        # everything above and below the ground is an obstacle
        mask_obstacle[dZ > 0] = True
        if reader.below_is_obstacle:
            mask_obstacle[dZ < -below_ok] = True

        mask_inlier = points_to_image(coords[mask_inlier, :], reader)
        mask_obstacle = points_to_image(coords[mask_obstacle, :], reader)
        mask_ignore = points_to_image(coords[mask_ignore, :], reader)

        mask_final = np.zeros(shape=(config.color_height, config.color_width, 1), dtype=np.int32)
        mask_final[mask_obstacle == 1] = 2
        mask_final[mask_inlier == 1] = 1
        mask_final[mask_ignore == 1] = 0

        # recalculate the ignore mask to be all not colored pixels
        mask_ignore = mask_final == 0

        # border always has some artifacts
        mask_final[:2, :] = 0
        mask_final[-2:, :] = 0
        mask_final[:, :2] = 0
        mask_final[:, -2:] = 0

        # DEBUG: show mask before crf
        if config.debug_step5:
            cv2.imshow("mask", mask_final.astype(np.uint8) * 127)
            cv2.waitKey(0)

        # DEBUG: set to false for no crf
        if config.crf_enabled:
            d = dcrf.DenseCRF2D(config.color_width, config.color_height, 2)
            U = unary_from_labels(mask_final[..., 0], 2, gt_prob=config.crf_label_gt_prob, zero_unsure=True)
            d.setUnaryEnergy(U)

            config.configure_crf(d, color)

            Q = d.inference(config.crf_inference_count)
            MAP = np.argmax(Q, axis=0)
            MAP = (MAP.reshape(color.shape[0], color.shape[1]).astype(np.uint8))[..., np.newaxis]
        else:
            MAP = mask_final.astype(np.uint8)

        MAP[mask_ignore == 1] = 255

        # DEBUG: show final mask in gray
        if config.debug_step5:
            cv2.imshow("mask", MAP * 127)
            cv2.waitKey(0)

        if reader.is_inverse:
            MAP = cv2.rotate(MAP, cv2.ROTATE_180)
            color = cv2.rotate(color, cv2.ROTATE_180)

        filename = os.path.basename(name) + "_%05d.png" % i
        write_png(MAP, os.path.join(name, "label", filename))
        cv2.imwrite(os.path.join(name, "color", filename), color)

        # DEBUG: show final color mask
        if config.debug_step5:
            label = Labels(name)
            cv2.imshow("final", cv2.addWeighted(label.color(i), 0.5, label.label(i), 0.5, 0))
            cv2.waitKey(0)


def calculate_all_masks():
    folders = glob.glob(os.path.join(config.data_directory, "**/planes.npy"))
    for folder in folders:
        dir = os.path.dirname(folder)
        if os.path.exists(dir + "/label") and "demo" not in dir:
            continue
        calculate_masks(dir)


if __name__ == "__main__":
    calculate_all_masks()
