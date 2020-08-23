import cv2
import numpy as np

np.random.seed(0)

class HOG(object):
    '''提取特征'''

    # Grayscale
    def BGR2GRAY(self, img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
        return gray

    def __init__(self, gray):
        if len(gray.shape) != 2:
            gray = self.BGR2GRAY(gray)
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

class Units(object):
    # get IoU overlap ratio
    def iou(sef, a, b):
        # get area of a
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        # get area of b
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        # get left top x of IoU
        iou_x1 = np.maximum(a[0], b[0])
        # get left top y of IoU
        iou_y1 = np.maximum(a[1], b[1])
        # get right bottom of IoU
        iou_x2 = np.minimum(a[2], b[2])
        # get right bottom of IoU
        iou_y2 = np.minimum(a[3], b[3])

        # get width of IoU
        iou_w = iou_x2 - iou_x1
        # get height of IoU
        iou_h = iou_y2 - iou_y1

        # get area of IoU
        area_iou = iou_w * iou_h
        # get overlap ratio between IoU and all area
        iou = area_iou / (area_a + area_b - area_iou)

        return iou

    # resize using bi-linear
    def resize(self, img, h, w):
        # get shape
        _h, _w, _c = img.shape

        # get resize ratio
        ah = 1. * h / _h
        aw = 1. * w / _w

        # get index of each y
        y = np.arange(h).repeat(w).reshape(w, -1)
        # get index of each x
        x = np.tile(np.arange(w), (h, 1))

        # get coordinate toward x and y of resized image
        y = (y / ah)
        x = (x / aw)

        # transfer to int
        ix = np.floor(x).astype(np.int32)
        iy = np.floor(y).astype(np.int32)

        # clip index
        ix = np.minimum(ix, _w - 2)
        iy = np.minimum(iy, _h - 2)

        # get distance between original image index and resized image index
        dx = x - ix
        dy = y - iy

        dx = np.tile(dx, [_c, 1, 1]).transpose(1, 2, 0)
        dy = np.tile(dy, [_c, 1, 1]).transpose(1, 2, 0)

        # resize
        out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (1 - dx) * dy * img[
            iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]
        out[out > 255] = 255

        return out

    # crop bounding box and make dataset
    def make_dataset(self, img, gt, Crop_N=200, L=60, th=0.5, H_size=32):
        # get shape
        H, W, _ = img.shape

        # get HOG feature dimension
        HOG_feature_N = ((H_size // 8) ** 2) * 9

        # prepare database
        db = np.zeros([Crop_N, HOG_feature_N + 1])

        # each crop
        for i in range(Crop_N):
            # get left top x of crop bounding box
            x1 = np.random.randint(W - L)
            # get left top y of crop bounding box
            y1 = np.random.randint(H - L)
            # get right bottom x of crop bounding box
            x2 = x1 + L
            # get right bottom y of crop bounding box
            y2 = y1 + L

            # get bounding box
            crop = np.array((x1, y1, x2, y2))

            _iou = np.zeros((3,))
            _iou[0] = self.iou(gt, crop)
            # _iou[1] = iou(gt2, crop)
            # _iou[2] = iou(gt3, crop)

            # get label
            if _iou.max() >= th:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                label = 1
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                label = 0

            # crop area
            crop_area = img[y1:y2, x1:x2]

            # resize crop area
            crop_area = self.resize(crop_area, H_size, H_size)

            # get HOG feature
            _hog = HOG(crop_area).run()

            # store HOG feature and label
            db[i, :HOG_feature_N] = _hog.ravel()
            db[i, -1] = label

        return db


# sliding window
def sliding_window(img, H_size=32):
    # get shape
    H, W, _ = img.shape

    # base rectangle [h, w]
    recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

    obj_units = Units()
    # sliding window
    for y in range(0, H, 4):
        for x in range(0, W, 4):
            for rec in recs:
                # get half size of ractangle
                dh = int(rec[0] // 2)
                dw = int(rec[1] // 2)

                # get left top x
                x1 = max(x - dw, 0)
                # get left top y
                x2 = min(x + dw, W)
                # get right bottom x
                y1 = max(y - dh, 0)
                # get right bottom y
                y2 = min(y + dh, H)

                # crop region
                region = img[max(y - dh, 0): min(y + dh, H), max(x - dw, 0): min(x + dw, W)]

                # resize crop region
                region = obj_units.resize(region, H_size, H_size)

                # get HOG feature
                region_hog = HOG(region).run().ravel()


# read detect target image
img = cv2.imread("imori_many.jpg")

sliding_window(img)