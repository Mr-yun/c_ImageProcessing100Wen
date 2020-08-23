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
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        iou_x1 = np.maximum(a[0], b[0])
        iou_y1 = np.maximum(a[1], b[1])
        iou_x2 = np.minimum(a[2], b[2])
        iou_y2 = np.minimum(a[3], b[3])
        iou_w = max(iou_x2 - iou_x1, 0)
        iou_h = max(iou_y2 - iou_y1, 0)
        area_iou = iou_w * iou_h
        iou = area_iou / (area_a + area_b - area_iou)
        return iou

    # resize using bi-linear
    def resize(self, img, h, w):
        # get shape
        if len(img.shape) > 2:
            _h, _w, _c = img.shape
        else:
            _h, _w = img.shape

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

        if len(img.shape) > 2:
            dx = np.tile(dx, [_c, 1, 1]).transpose(1, 2, 0)
            dy = np.tile(dy, [_c, 1, 1]).transpose(1, 2, 0)

        # resize
        out = (1 - dx) * (1 - dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix + 1] + (1 - dx) * dy * img[
            iy + 1, ix] + dx * dy * img[iy + 1, ix + 1]
        out[out > 255] = 255

        return out

    # crop bounding box and make dataset
    def make_dataset(self, img, gray, gt, Crop_N=200, L=60, th=0.5, H_size=32):
        # get shape
        if len(img.shape) > 2:
            H, W, _ = img.shape
        else:
            H, W = img.shape

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

            _iou = self.iou(gt, crop)

            # get label
            if _iou >= th:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                label = 1
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                label = 0

            # crop area
            crop_area = gray[y1:y2, x1:x2]

            # resize crop area
            crop_area = self.resize(crop_area, H_size, H_size)

            # get HOG feature
            _hog = HOG(crop_area).run()

            # store HOG feature and label
            db[i, :HOG_feature_N] = _hog.ravel()
            db[i, -1] = label

        return db


class NN(object):
    def __init__(self, ind=2, lr=0.01):
        self.nets = [
            [np.random.normal(0, 1, [ind, 64]),
             np.random.normal(0, 1, [64])],
            [np.random.normal(0, 1, [64, 64]),
             np.random.normal(0, 1, [64])],
            [np.random.normal(0, 1, [64, 1]),
             np.random.normal(0, 1, [1])],
        ]
        self.lr = lr  # 学习率

    def sigmoid(self, x):
        # sigmoid 激活函数
        return 1. / (1. + np.exp(-x))

    def forward(self, x):
        # 前向操作
        result = [x]
        for net in self.nets:
            result.append(
                self.sigmoid(np.dot(result[-1], net[0]) + net[1])
            )
        self.result = result
        return self.result[-1]

    def train(self, t):
        # 反馈
        grad_u = None
        len_net = len(self.nets)
        result = self.result[::-1]

        for i, v in enumerate(result[:-1]):
            if grad_u is None:
                grad_u = (v - t) * v * (1 - v)
            else:
                grad_u = np.dot(grad_u, self.nets[len_net - i][0].T) * v * (1 - v)

            grad_w = np.dot(result[i + 1].T, grad_u)
            grad_b = np.dot(np.ones([grad_u.shape[0]]), grad_u)

            self.nets[len_net - 1 - i][0] -= self.lr * grad_w
            self.nets[len_net - 1 - i][1] -= self.lr * grad_b


class Detect(object):
    # sliding window
    def sliding_window(self, nn, img, gray_img, H_size=32):
        # get shape
        H, W, _ = img.shape

        # base rectangle [h, w]
        recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)
        detects = np.ndarray((0, 5), dtype=np.float32)

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
                    region = gray_img[max(y - dh, 0): min(y + dh, H), max(x - dw, 0): min(x + dw, W)]

                    # resize crop region
                    region = obj_units.resize(region, H_size, H_size)

                    # get HOG feature
                    region_hog = HOG(region).run().ravel()

                    score = nn.forward(region_hog)
                    if score >= 0.7:
                        detects = np.vstack((detects, np.array((x1, y1, x2, y2, score))))

        # print(len(detects))
        # print(detects)
        return detects

    # Non-maximum suppression
    def nms(self, _bboxes, iou_th):
        bboxes = _bboxes.copy()

        # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # 宽
        # bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # 高

        # 自信度由高到底
        sort_inds = np.argsort(bboxes[:, -1])[::-1]
        return_inds = []

        unselected_inds = sort_inds.copy()

        while len(unselected_inds) > 0:
            process_bboxes = bboxes[unselected_inds]
            argmax_score_ind = np.argmax(process_bboxes[::, -1])
            max_score_ind = unselected_inds[argmax_score_ind]
            return_inds.append(max_score_ind)
            unselected_inds = np.delete(unselected_inds, argmax_score_ind)

            base_bbox = bboxes[max_score_ind]
            list_compare_bboxes = bboxes[unselected_inds]

            base_area, iou_area = self.get_iou(base_bbox, list_compare_bboxes)

            compare_w = np.maximum(list_compare_bboxes[:, 2], 0) - np.maximum(list_compare_bboxes[:, 0], 0)
            compare_h = np.maximum(list_compare_bboxes[:, 3], 0) - np.maximum(list_compare_bboxes[:, 1], 0)
            compare_area = compare_w * compare_h

            # bbox's index which iou ratio over threshold is excluded
            all_area = compare_area + base_area - iou_area
            iou_ratio = np.zeros((len(unselected_inds)))
            iou_ratio[all_area < 0.9] = 0.  # 存在一定交集才算
            _ind = all_area >= 0.9
            iou_ratio[_ind] = iou_area[_ind] / all_area[_ind]

            unselected_inds = np.delete(unselected_inds, np.where(iou_ratio >= iou_th)[0])

        return return_inds

    def get_iou(self, base_bbox, list_compare_bboxes):
        base_area = np.maximum(base_bbox[2] - base_bbox[0], 0) * np.maximum(base_bbox[3] - base_bbox[1], 0)

        # compute iou-area between base bbox and other bboxes
        iou_x1 = np.maximum(base_bbox[0], list_compare_bboxes[:, 0])
        iou_y1 = np.maximum(base_bbox[1], list_compare_bboxes[:, 1])
        iou_x2 = np.minimum(base_bbox[2], list_compare_bboxes[:, 2])
        iou_y2 = np.minimum(base_bbox[3], list_compare_bboxes[:, 3])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area = iou_w * iou_h
        return base_area, iou_area


def select_head():
    img = cv2.imread("imori_1.jpg")
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    gt = np.array((47, 41, 129, 103), dtype=np.float32)
    cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 255), 1)
    return gt, gray, img


def train_nets(db):
    nn = NN(ind=db.shape[1] - 1, lr=0.01)
    for i in range(10000):
        nn.forward(db[:, :db.shape[1] - 1])
        nn.train(db[:, -1][..., None])
    return nn


def pred():
    obj_detect = Detect()
    img2 = cv2.imread("imori_many.jpg")
    gray2 = 0.2126 * img2[..., 2] + 0.7152 * img2[..., 1] + 0.0722 * img2[..., 0]
    detects = obj_detect.sliding_window(nn, img2, gray2)  # [(上左点坐标,下右点坐标,执行度)...]
    nms_detects = obj_detect.nms(detects, iou_th=0.25)

    for d in nms_detects:
        v = list(map(int, detects[d][:4]))
        cv2.rectangle(img2, (v[0], v[1]), (v[2], v[3]), (0, 0, 255), 1)
        cv2.putText(img2, "{:.2f}".format(detects[d][-1][0]), (v[0], v[1] + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    cv2.imshow("result", img2)
    cv2.waitKey(0)


gt, gray, img = select_head()
db = Units().make_dataset(img, gray, gt)
nn = train_nets(db)
pred()
