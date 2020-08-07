# -*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_gray(img):
    out = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return out.astype(np.uint8)


def get_otsu(gray):
    max_sigma = 0
    max_t = 0

    gray_range = gray.shape[0] * gray.shape[1]

    for _t in range(1, 255):
        v0 = gray[np.where(gray < _t)[0]]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / gray_range

        v1 = gray[np.where(gray >= _t)[0]]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / gray_range

        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t
    gray[gray < max_t] = 0
    gray[gray >= max_t] = 255

    return gray


def morphology(otsu, out, color, morphology_time=1):
    MF = np.array(((0, 1, 0),
                   (1, 0, 1),
                   (0, 1, 0)), dtype=np.int)

    H, W = (otsu.shape[0], otsu.shape[1])

    for i in range(morphology_time):
        tmp = np.pad(otsu, (1, 1), 'edge')
        for y in range(1, H + 1):
            for x in range(1, W + 1):
                if color == 0:
                    if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) < 255 * 4:
                        out[y - 1, x - 1] = color
                else:
                    if np.sum(MF * tmp[y - 1:y + 2, x - 1:x + 2]) >= 255:
                        out[y - 1, x - 1] = color
    return out


if __name__ == "__main__":
    img = cv2.imread("imori.jpg").astype(np.float32)
    H, W, C = img.shape

    gray = get_gray(img)
    otsu = get_otsu(gray)

    erode = morphology(otsu.copy(), otsu.copy(), 0, morphology_time=3)
    dilate = morphology(erode, erode, 255, morphology_time=3)

    out = otsu - dilate
    cv2.imshow("dilate", dilate)
    cv2.imshow("result", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
