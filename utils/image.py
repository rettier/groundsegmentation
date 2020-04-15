import cv2
import numpy as np

red = (0, 0, 1.)
green = (0, 1., 0)
blue = (1., 0, 0)


def blend(image1, mask, color, opacity):
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    mask_color = np.ndarray(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    for x in range(mask_color.shape[-1]):
        mask_color[..., x] = mask
    mask_color *= color
    return cv2.addWeighted(image1, (1. - opacity), (mask_color * 255).astype(np.uint8), opacity, 0)


def dice(y_true, prediction, ignore_area):
    ignore_mask = ignore_area != 0
    number_pixels = np.count_nonzero(ignore_mask == 0)
    y_true = y_true.copy()
    prediction = prediction.copy()
    y_true[ignore_mask] = 99
    prediction[ignore_mask] = 98

    intersection0 = np.logical_and(y_true == 0, prediction == 0)
    union0 = np.logical_or(y_true == 0, prediction == 0)

    intersection1 = np.logical_and(y_true == 1, prediction == 1)
    union1 = np.logical_or(y_true == 1, prediction == 1)

    iou0 = np.sum(intersection0) / np.sum(union0)
    iou1 = np.sum(intersection1) / np.sum(union1)

    # while True:
    #    cv2.imshow("iou", intersection1.astype(np.float32))
    #    cv2.waitKey(0)
    #    cv2.imshow("iou", union1.astype(np.float32))
    #    cv2.waitKey(0)

    return iou0, iou1, number_pixels / (1.0 * (y_true.shape[0] * y_true.shape[1]))
