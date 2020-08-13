# -*- coding:utf-8 -*-
import numpy as np
import cv2

class HOG(object):
    def __init__(self, gray):
        if len(gray.shape) != 2:
            raise ValueError("灰度图像")
        self.gray = gray

    def get_gradXY(self):
        H, W = self.gray.shape
        gray = np.pad(self.gray, (1, 1), 'edge')

        gx = gray[1:H + 1, 2:] - gray[1:H + 1, :W]
        gy = gray[2:, 1:W + 1] - gray[:H, 1:W + 1]

        gx[gx == 0] = 1e-6

        return gx, gy

    def get_MagGrad(self, gx, gy):
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        gradient = np.arctan(gy / gx)

        # todo 小于0 ？？？
        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    def quantization(self, gradient):
        # prepare quantization tablec
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized

    def gradient_histogram(self, gradient_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x,
                                  gradient_quantized[y * 4 + j, x * 4 + i]] += \
                            magnitude[y * 4 + j, x * 4 + i]

        return histogram

    def normalization(self, histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape

        ## each histogram
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                # for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0): min(y + 2, cell_N_H),
                                                  max(x - 1, 0): min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram

    def run(self):
        # 1. Gray -> Gradient x and y
        gx, gy = self.get_gradXY()

        # 2. get gradient magnitude and angle
        magnitude, gradient = self.get_MagGrad(gx, gy)

        # 3. Quantization
        gradient_quantized = self.quantization(gradient)

        # 4. Gradient histogram
        histogram = self.gradient_histogram(gradient_quantized, magnitude)

        # 5. Histogram normalization
        histogram = self.normalization(histogram)

        return histogram


def draw_HOG(img, histogram):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    def draw(gray, histogram, N=8):
        # get shape
        H, W = gray.shape
        cell_N_H, cell_N_W, _ = histogram.shape

        ## Draw
        out = gray[1: H + 1, 1: W + 1].copy().astype(np.uint8)

        for y in range(cell_N_H):
            for x in range(cell_N_W):
                cx = x * N + N // 2
                cy = y * N + N // 2
                x1 = cx + N // 2 - 1
                y1 = cy
                x2 = cx - N // 2 + 1
                y2 = cy

                h = histogram[y, x] / np.sum(histogram[y, x])
                h /= h.max()

                for c in range(9):
                    # angle = (20 * c + 10 - 90) / 180. * np.pi
                    # get angle
                    angle = (20 * c + 10) / 180. * np.pi
                    rx = int(np.sin(angle) * (x1 - cx) + np.cos(angle) * (y1 - cy) + cx)
                    ry = int(np.cos(angle) * (x1 - cx) - np.cos(angle) * (y1 - cy) + cy)
                    lx = int(np.sin(angle) * (x2 - cx) + np.cos(angle) * (y2 - cy) + cx)
                    ly = int(np.cos(angle) * (x2 - cx) - np.cos(angle) * (y2 - cy) + cy)

                    # color is HOG value
                    c = int(255. * h[c])

                    # draw line
                    cv2.line(out, (lx, ly), (rx, ry), (c, c, c), thickness=1)

        return out

    # get gray
    gray = BGR2GRAY(img)

    # draw HOG
    out = draw(gray, histogram)

    return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# get HOG
obj_hog = HOG(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY))
histogram = obj_hog.run()

# draw HOG
out = draw_HOG(img, histogram)


# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()