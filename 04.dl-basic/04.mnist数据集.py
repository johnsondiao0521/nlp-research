# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:johnsondiao
@File: 04.mnist数据集.py
@Time: 2023-04-08 10:30
@Desc:
"""
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def load_images(path):
    with open(path, "rb") as f:
        data = f.read()

    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, 28, 28)


def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()
    return np.asarray(bytearray(data[8:]), dtype=np.int32)


if __name__ == "__main__":
    train_images = load_images(os.path.join("data", "mnist", "train-images.idx3-ubyte"))
    train_labels = load_labels(os.path.join("data", "mnist", "train-labels.idx1-ubyte"))

    print(train_images.shape)
    print(train_labels.shape)