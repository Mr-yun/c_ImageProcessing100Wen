# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Grayscale
def BGR2GRAY(img):
    # Grayscale
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    return gray


# Gabor
def Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    '''
    :param K_size: ->y、x取[−k//2,k//2]
    :param Sigma:  Gabor 滤波器的椭圆度
    :param Gamma: 高斯分布的标准差
    :param Lambda:  波长
    :param Psi: 相位
    :param angle: 滤波核中平行条带的方向
    :return:
    '''
    # get half size
    d = K_size // 2
    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            px = x - d
            py = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            _x = np.cos(theta) * px + np.sin(theta) * py
            _y = -np.sin(theta) * px + np.cos(theta) * py

            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) / (2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)
    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor


def Gabor_filtering(gray, K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
    H, W = gray.shape

    gray = np.pad(gray, (K_size // 2, K_size // 2), 'edge')

    out = np.zeros((H, W), dtype=np.float32)

    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)



    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y:y + K_size, x:x + K_size] * gabor)

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


def Gabor_process(img):
    H, W, _ = img.shape

    # gray scale
    gray = BGR2GRAY(img).astype(np.float32)

    As = [0, 45, 90, 135]

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

    out = np.zeros([H, W], dtype=np.float32)

    # each angle
    for i, A in enumerate(As):
        # gabor filtering
        _out = Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, angle=A)

        # add gabor filtered image
        out += _out

        # scale normalization
    out = out / out.max() * 255
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# gabor process
out = Gabor_process(img)


cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
