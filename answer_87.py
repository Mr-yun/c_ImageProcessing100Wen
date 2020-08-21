# -*- coding:utf-8 -*-
from glob import glob

import cv2
import numpy as np


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
            db[i, j + 4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j + 8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

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
def test_DB(db, pdb, N=3):
    # get test image path
    test = glob("dataset/test_*")
    test.sort()

    accuracy_N = 0.

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

        # 选取最相识的N个
        pred_i = np.argsort(difs)[:N]

        # predict class index
        pred = db[pred_i, -1]

        # get class label
        if len(pred[pred == 0]) > len(pred[pred == 1]):
            pl = "akahara"
        else:
            pl = 'madara'

        print(path, "is similar >> ", end='')
        for i in pred_i:
            print(pdb[i], end=', ')
        print("|Pred >>", pl)

        # count accuracy
        gt = "akahara" if "akahara" in path else "madara"
        if gt == pl:
            accuracy_N += 1.

    accuracy = accuracy_N / len(test)
    print("Accuracy >>", accuracy, "({}/{})".format(int(accuracy_N), len(test)))


db, pdb = get_DB()
test_DB(db, pdb)
