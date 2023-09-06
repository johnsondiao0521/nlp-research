"""
一条一条数据的求loss,然后加上验证集
"""
import numpy as np
import os
import struct


def load_images(file):
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, 28, 28)


def load_labels(file):
    with open(file, "rb") as f:
        data = f.read()

    return np.asarray(bytearray(data[8:]), dtype=np.int32)


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex)

    return ex / sum_ex


def make_onehot(labels, class_num):
    result = np.zeros((labels.shape[0], class_num))

    for idx, cls in enumerate(labels):
        result[idx][cls] = 1
    return result


if __name__ == '__main__':
    train_images = load_images(os.path.join("data", "mnist", "train-images.idx3-ubyte")) / 255
    train_labels = load_labels(os.path.join("data", "mnist", "train-labels.idx1-ubyte"))

    dev_images = load_images(os.path.join("data", "mnist", "t10k-images.idx3-ubyte")) / 255
    dev_labels = load_labels(os.path.join("data", "mnist", "t10k-labels.idx1-ubyte"))

    train_labels = make_onehot(train_labels, 10)

    train_images = train_images.reshape(60000, 784)
    dev_images = dev_images.reshape(-1, 784)

    w = np.random.normal(0, 1, size=(784, 10))
    b = np.random.normal(0, 1, size=(1, 10))

    epoch = 100
    lr = 0.01

    for e in range(epoch):
        for idx in range(60000):
            image = train_images[idx: idx+1]
            label = train_labels[idx: idx+1]

            pre = image @ w + b

            p = softmax(pre)
            loss = -np.sum(label * np.log(p))

            G = p - label

            delta_w = image.T @ G
            delta_b = G

            w -= lr * delta_w
            b -= lr * delta_b

        right_num = 0
        for idx in range(len(dev_images)):
            image = dev_images[idx: idx + 1]
            label = dev_labels[idx]

            pre = image @ w + b
            pre_idx = int(np.argmax(pre, axis=1))
            right_num += int(pre_idx == label)

        acc = right_num / len(dev_images)
        print(f"acc:{acc:.3f}")



