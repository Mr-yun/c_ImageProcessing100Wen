# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

        # show result
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)

        plt.subplot(1, 3, 1)
        plt.imshow(Ix2, cmap='gray')
        plt.title("Ix^2")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(Iy2, cmap='gray')
        plt.title("Iy^2")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(Ixy, cmap='gray')
        plt.title("Ixy")
        plt.axis("off")

        plt.savefig("out.png")
        plt.show()



# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)

# Hessian corner detection
Hessian_corner().run(img)

