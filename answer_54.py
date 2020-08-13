# -*- coding:utf-8 -*-
import cv2
import numpy as np

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

tmp = cv2.imread("imori_part.jpg").astype(np.float32)
Ht, Wt, Ct = tmp.shape

i, j = -1, -1
v = 255 * H * W * C
for y in range(H - Ht):
    for x in range(W - Wt):
        _v = np.sum((img[y:y + Ht, x:x + Wt] - tmp) ** 2)
        if _v < v:
            v = _v
            i, j = x, y

cv2.rectangle(img, pt1=(i, j), pt2=(i + Wt, j + Ht), color=(0, 0, 255), thickness=1)

cv2.imshow("result", img.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()