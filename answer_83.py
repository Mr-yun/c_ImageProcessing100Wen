# -*- coding:utf-8 -*-
import cv2
import numpy as np


# Hessian corner detection
class Hessian_corner():

    ## Grayscale
    def BGR2GRAY(self, img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        gray = gray.astype(np.uint8)
        return gray

    ## Sobel
    def Sobel_filtering(self, gray):
        H, W = gray.shape

        sobelx = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
        sobely = np.array([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
        tmp = np.pad(gray, (1, 1), 'edge')
        ix = np.zeros((H, W), dtype=np.float32)
        iy = np.zeros((H, W), dtype=np.float32)

        for h in range(H):
            for w in range(W):
                ix[h, w] = np.mean(tmp[h:h + 3, w:w + 3] * sobelx)
                iy[h, w] = np.mean(tmp[h:h + 3, w:w + 3] * sobely)
        Ix2 = ix * ix
        Iy2 = iy * iy
        Ixy = ix * iy
        return Ix2, Iy2, Ixy

    ## Hessian
    def gaussian_filtering(self, I, k_size=3, sigma=3):
        H, W = I.shape

        I_t = np.pad(I, (k_size // 2, k_size // 2), 'edge')

        k = np.zeros((k_size, k_size), dtype=np.float)

        # 高斯卷积核
        for x in range(k_size):
            _x = x - k_size // 2
            for y in range(k_size):
                _y = y - k_size // 2
                k[y, x] = np.exp(-(_x * _x + _y * _y) / (2 * (sigma ** 2)))
        k /= (sigma * np.sqrt(2 * np.pi))
        k /= k.sum()

        for y in range(H):
            for x in range(W):
                I[y, x] = np.sum(I_t[y: y + k_size, x: x + k_size] * k)

        return I

    ## Hessian
    def corner_detect(self, gray, Ix2, Iy2, Ixy):
        H, W = gray.shape

        out = np.array((gray, gray, gray))
        out = np.transpose(out, (1, 2, 0))

        Hes = np.zeros((H, W))
        for h in range(H):
            for w in range(W):
                Hes[h, w] = Ix2[h, w] * Iy2[h, w] - Ixy[h, w] ** 2

        max_hes = np.max(Hes) * 0.1

        for y in range(H):
            for x in range(W):
                # 当前像素式临近橡像素的最大点，同时大于det最大值×0.1
                if Hes[y, x] == np.max(Hes[max(y - 1, 0): min(y + 2, H), max(x - 1, 0): min(x + 2, W)]) and Hes[
                    y, x] > max_hes:
                    out[y, x] = [0, 0, 255]
        return out

    def run(self, img):

        # 1. grayscale
        gray = self.BGR2GRAY(img)

        # 3. corner detection
        out = self.Sobel_filtering(gray)

        # 2. get difference image
        Ix2, Iy2, Ixy = self.Sobel_filtering(gray)

        # 3. gaussian filtering
        Ix2 = self.gaussian_filtering(Ix2, k_size=3, sigma=3)
        Iy2 = self.gaussian_filtering(Iy2, k_size=3, sigma=3)
        Ixy = self.gaussian_filtering(Ixy, k_size=3, sigma=3)

        # 3. corner detection
        out = self.corner_detect(gray, Ix2, Iy2, Ixy)

        return out


# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Hessian corner detection
out = Hessian_corner().run(img)

cv2.imshow("result", out)
cv2.waitKey(0)
