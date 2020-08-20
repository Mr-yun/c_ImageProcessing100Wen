# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

## 图片颜色分为个区域
def dic_color(img):
    img //= 63
    img = img * 64 + 32
    return img

## Database
def get_DB():

    # dataset目录下查找已train_开头的文件
    train = glob("dataset/train_*")
    train.sort()

    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32)

    pdb = []

    # each image
    for i, path in enumerate(train):
        img = dic_color(cv2.imread(path))
        # get histogram
        for j in range(4):
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

            # get class
            if 'akahara' in path:
                cls = 0
            elif 'madara' in path:
                cls = 1

            # store class label
            db[i, -1] = cls

            # store image path
            pdb.append(path)

    return db, pdb


# test
def test_DB(db, pdb):
    # get test image path
    test = glob("dataset/test_*")
    test.sort()

    success_num = 0.

    # each image
    for path in test:
        # read image
        img = dic_color(cv2.imread(path))

        # get histogram
        hist = np.zeros(12, dtype=np.int32)
        for j in range(4):
            hist[j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            hist[j + 4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            hist[j + 8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # 最临近法（在训练集中找出直方图总和最小的，即为最匹配类）
        difs = np.abs(db[:, :12] - hist)
        difs = np.sum(difs, axis=1)

        # get argmin of difference
        pred_i = np.argmin(difs)

        # get prediction label
        pred = db[pred_i, -1]

        if pred == 0:
            pl = "akahara"
        elif pred == 1:
            pl = "madara"

        print(path, "is similar >>", pdb[pred_i], " Pred >>", pl)


db, pdb = get_DB()
test_DB(db, pdb)