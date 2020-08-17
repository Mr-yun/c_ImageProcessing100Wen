# -*- coding:utf-8 -*-
import cv2
import numpy as np


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

            gabor[y, x] = np.exp(-(_x ** 2 + Gamma ** 2 * _y ** 2) /( 2 * Sigma ** 2)) * np.cos(
                2 * np.pi * _x / Lambda + Psi)
    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor

As = [0, 45, 90, 135]
# each angle
for i, A in enumerate(As):
    # get gabor kernel
    gabor = Gabor_filter(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=A)

    out = gabor - np.min(gabor)
    out /= np.max(out)
    out *= 255

    out = out.astype(np.uint8)
    cv2.imshow("{}".format(i), out)
cv2.waitKey(0)
