# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:johnsondiao
@File: 04.手写数字识别_1.py
@Time: 2023-04-18 15:16
@Desc: 一条一条数据的求loss
"""

import numpy as np
import os
import struct


def load_images(file):
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)


def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    return np.asarray(bytearray(data[8:]), dtype=np.uint32)


def make_onehot(labels, cls_num):  # 用一个数值代表类别不合适，用向量代表标签类型
    result = np.zeros((labels.shape[0], cls_num))
    for idx, cls_ in enumerate(labels):
        result[idx][cls_] = 1
    return result


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex)
    result = ex / sum_ex
    return result


if __name__ == '__main__':
    train_images = load_images(os.path.join("data", "mnist", "train-images.idx3-ubyte")) / 255
    train_labels = load_labels(os.path.join("data", "mnist", "train-labels.idx1-ubyte"))
    train_labels = make_onehot(train_labels, 10)

    w = np.random.normal(0, 1, size=(784, 10))
    b = np.random.normal(0, 1, size=(1, 10))

    epoch = 10
    lr = 0.001

    for e in range(epoch):
        for idx in range(60000):
            image = train_images[idx: idx + 1]
            label = train_labels[idx: idx + 1]

            pre = image @ w + b
            p = softmax(pre)
            loss = - np.sum(label * np.log(p))
            G = p - label

            delta_w = image.T @ G
            delta_b = G

            w -= lr * delta_w
            b -= lr * delta_b

        print(loss)


