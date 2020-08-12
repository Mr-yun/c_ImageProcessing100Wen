import cv2
import numpy as np
from time import time

# x4 , x3 , x2
# x5 , x0 , x1
# x6 , x7 , x8
# 背景值為0，前景值為1
def condition_1(out, tmp, x, y, judge):
    H, W = out.shape
    _tmp = 1 - tmp

    # 4− 近邻中有至少一个0
    if (tmp[y, min(x + 1, W - 1)] * tmp[max(y - 1, 0), x] * tmp[y, max(x - 1, 0)] * tmp[
        min(y + 1, H - 1), x]) == 0:
        judge += 1

    c = 0
    c += (_tmp[y, min(x + 1, W - 1)] - _tmp[y, min(x + 1, W - 1)] * _tmp[max(y - 1, 0), min(x + 1, W - 1)] *
          _tmp[max(y - 1, 0), x])
    c += (_tmp[max(y - 1, 0), x] - _tmp[max(y - 1, 0), x] * _tmp[max(y - 1, 0), max(x - 1, 0)] * _tmp[
        y, max(x - 1, 0)])
    c += (_tmp[y, max(x - 1, 0)] - _tmp[y, max(x - 1, 0)] * _tmp[min(y + 1, H - 1), max(x - 1, 0)] * _tmp[
        min(y + 1, H - 1), x])
    c += (_tmp[min(y + 1, H - 1), x] - _tmp[min(y + 1, H - 1), x] * _tmp[
        min(y + 1, H - 1), min(x + 1, W - 1)] * _tmp[y, min(x + 1, W - 1)])
    # 8−连接数为1
    if c == 1:
        judge += 1

    # condition 3
    if np.sum(tmp[max(y - 1, 0): min(y + 2, H), max(x - 1, 0): min(x + 2, W)]) >= 3:
        judge += 1

    # x1至x8的绝对值之和大于2
    if np.sum(out[max(y - 1, 0): min(y + 2, H), max(x - 1, 0): min(x + 2, W)]) >= 2:
        judge += 1

    return judge


def condition_2(out, tmp, x, y, judge):
    # 由于代码是至上而下,至左而右
    # 所以判断上一个细化操作(H*W)和本次操作后8像素块变化,只能比较 上一排3个,以及左边一个,因为其他的还未执行到
    H, W = out.shape
    _tmp2 = 1 - out

    ## condition 5

    if out[max(y - 1, 0), max(x - 1, 0)] != tmp[max(y - 1, 0), max(x - 1, 0)]:
        judge += 1
    else:
        c = 0
        c += (_tmp2[y, min(x + 1, W - 1)] - _tmp2[y, min(x + 1, W - 1)] * _tmp2[
            max(y - 1, 0), min(x + 1, W - 1)] * _tmp2[max(y - 1, 0), x])
        c += (_tmp2[max(y - 1, 0), x] - _tmp2[max(y - 1, 0), x] * (1 - tmp[max(y - 1, 0), max(x - 1, 0)]) *
              _tmp2[y, max(x - 1, 0)])
        c += (_tmp2[y, max(x - 1, 0)] - _tmp2[y, max(x - 1, 0)] * _tmp2[min(y + 1, H - 1), max(x - 1, 0)] *
              _tmp2[min(y + 1, H - 1), x])
        c += (_tmp2[min(y + 1, H - 1), x] - _tmp2[min(y + 1, H - 1), x] * _tmp2[
            min(y + 1, H - 1), min(x + 1, W - 1)] * _tmp2[y, min(x + 1, W - 1)])
        if c == 1:  # x4 左上
            judge += 1
        else:
            return judge
    if out[max(y - 1, 0), x] != tmp[max(y - 1, 0), x]:
        judge += 1
    else:
        c = 0
        c += (_tmp2[y, min(x + 1, W - 1)] - _tmp2[y, min(x + 1, W - 1)] * _tmp2[
            max(y - 1, 0), min(x + 1, W - 1)] * (1 - tmp[max(y - 1, 0), x]))
        c += ((1 - tmp[max(y - 1, 0), x]) - (1 - tmp[max(y - 1, 0), x]) * _tmp2[max(y - 1, 0), max(x - 1, 0)] *
              _tmp2[y, max(x - 1, 0)])
        c += (_tmp2[y, max(x - 1, 0)] - _tmp2[y, max(x - 1, 0)] * _tmp2[min(y + 1, H - 1), max(x - 1, 0)] *
              _tmp2[min(y + 1, H - 1), x])
        c += (_tmp2[min(y + 1, H - 1), x] - _tmp2[min(y + 1, H - 1), x] * _tmp2[
            min(y + 1, H - 1), min(x + 1, W - 1)] * _tmp2[y, min(x + 1, W - 1)])
        if c == 1:  # x3 正上
            judge += 1
        else:
            return judge

    if out[max(y - 1, 0), min(x + 1, W - 1)] != tmp[max(y - 1, 0), min(x + 1, W - 1)]:
        judge += 1
    else:
        c = 0
        c += (_tmp2[y, min(x + 1, W - 1)] - _tmp2[y, min(x + 1, W - 1)] * (
                1 - tmp[max(y - 1, 0), min(x + 1, W - 1)]) * _tmp2[max(y - 1, 0), x])
        c += (_tmp2[max(y - 1, 0), x] - _tmp2[max(y - 1, 0), x] * _tmp2[max(y - 1, 0), max(x - 1, 0)] * _tmp2[
            y, max(x - 1, 0)])
        c += (_tmp2[y, max(x - 1, 0)] - _tmp2[y, max(x - 1, 0)] * _tmp2[min(y + 1, H - 1), max(x - 1, 0)] *
              _tmp2[min(y + 1, H - 1), x])
        c += (_tmp2[min(y + 1, H - 1), x] - _tmp2[min(y + 1, H - 1), x] * _tmp2[
            min(y + 1, H - 1), min(x + 1, W - 1)] * _tmp2[y, min(x + 1, W - 1)])
        if c == 1 :  # x2 右上
            judge += 1
        else:
            return judge

    if out[y, max(x - 1, 0)] != tmp[y, max(x - 1, 0)]:
        judge += 1
    else:
        c = 0
        c += (_tmp2[y, min(x + 1, W - 1)] - _tmp2[y, min(x + 1, W - 1)] * _tmp2[
            max(y - 1, 0), min(x + 1, W - 1)] * _tmp2[max(y - 1, 0), x])
        c += (_tmp2[max(y - 1, 0), x] - _tmp2[max(y - 1, 0), x] * _tmp2[max(y - 1, 0), max(x - 1, 0)] * (
                1 - tmp[y, max(x - 1, 0)]))
        c += ((1 - tmp[y, max(x - 1, 0)]) - (1 - tmp[y, max(x - 1, 0)]) * _tmp2[
            min(y + 1, H - 1), max(x - 1, 0)] * _tmp2[min(y + 1, H - 1), x])
        c += (_tmp2[min(y + 1, H - 1), x] - _tmp2[min(y + 1, H - 1), x] * _tmp2[
            min(y + 1, H - 1), min(x + 1, W - 1)] * _tmp2[y, min(x + 1, W - 1)])
        if c == 1 :  # x4 左边
            judge += 1
        else:
            return judge

    return judge


# hilditch thining
def hilditch(img):
    # get shape
    H, W, C = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    # inverse pixel value
    count = 1
    while count > 0:
        count = 0
        tmp = out.copy()

        # each pixel
        for y in range(H):
            for x in range(W):
                # skip black pixel
                if out[y, x] < 1:
                    continue

                judge = 0
                judge = condition_1(out, tmp, x, y, judge)
                judge = condition_2(out, tmp, x, y, judge)

                if judge >= 8:
                    out[y, x] = 0
                    count += 1

    out = out.astype(np.uint8) * 255

    return out

# Read image
img = cv2.imread("gazo.png").astype(np.float32)

# hilditch thining
out = hilditch(img)

# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
