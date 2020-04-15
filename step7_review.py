import glob
import numpy as np
import os
from types import SimpleNamespace

import cv2

import config
from utils.reader import Labels

state = None


def reset_state(dir):
    global state
    labels = Labels(dir)
    good, bad = labels.load_good_bad_index()
    state = SimpleNamespace(
        i=None,
        history=[],
        labels=labels,
        good=good,
        bad=bad,
        ignore=good | bad,
        todo=None,
        mask_mode=0
    )
    state.todo = list(set(labels.names()) - state.ignore)


def name():
    return state.history[state.i]


def show_frame():
    if state.mask_mode == 0:
        img = cv2.addWeighted(state.labels.color(name=name()), 0.5,
                              state.labels.label(name=name(), cleaned=True), 0.5, 0)
    elif state.mask_mode == 1:
        img = state.labels.color(name=name())
    elif state.mask_mode == 2:
        img = state.labels.label(name=name(), cleaned=True)

    render = np.zeros(shape=(img.shape[0] + 20, img.shape[1], 3), dtype=np.uint8)
    render[20:, :, :] = img

    if name() in state.good:
        cv2.putText(render, "good", (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    elif name() in state.bad:
        cv2.putText(render, "bad", (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    else:
        cv2.putText(render, "not reviewed", (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    cv2.putText(render, name(), (300, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    cv2.imshow("Mask Review", render)
    return cv2.waitKey(0)


def forward():
    if state.i == len(state.history) - 1:
        take_next_frame()
    else:
        state.i += 1


def backward():
    if state.i == 0:
        return

    state.i -= 1


def good():
    state.good.add(name())
    state.bad.discard(name())


def bad():
    state.bad.add(name())
    state.good.discard(name())


def end():
    state.labels.save_good_bad_index(state.good, state.bad)
    state.todo = []


def process_key_stroke(key):
    key = key & 0xFF

    # right
    if key == 83:
        forward()

    # left
    elif key == 81:
        backward()

    # backspace
    elif key == 8:
        bad()
        forward()

    # return
    elif key == 10:
        good()
        forward()

    # escape
    elif key == 27:
        end()
        exit(0)

    elif key in [49, 50, 51]:
        state.mask_mode = key - 49


def take_next_frame():
    next_name = state.todo.pop()
    if next_name == "END":
        end()

    state.history.append(next_name)
    state.i = len(state.history) - 1


def review(dir):
    reset_state(dir)
    if not state.todo:
        return

    state.todo.insert(0, "END")
    take_next_frame()
    while state.todo:
        if name() in state.ignore:
            continue

        key = show_frame()
        process_key_stroke(key)

    end()


def review_all():
    folders = glob.glob(os.path.join(config.data_directory, "**/label_clean"))
    for folder in folders:
        review(os.path.dirname(folder))


if __name__ == "__main__":
    review_all()
