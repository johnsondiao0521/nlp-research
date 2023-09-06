"""
基于 手写数字识别_4, 实现2层linear
待解决：exp log 爆炸的问题
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


def make_onehot(labels, class_num):
    result = np.zeros((labels.shape[0], class_num))

    for idx, cls in enumerate(labels):
        result[idx][cls] = 1
    return result


class Dataset:
    def __init__(self, all_images, all_labels):
        self.all_images = all_images
        self.all_labels = all_labels

    def __getitem__(self, index):
        image = self.all_images[index]
        label = self.all_labels[index]

        return image, label

    def __len__(self):
        return len(self.all_labels)


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.cursor = 0

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration

        batch_images = []
        batch_labels = []
        for i in range(self.batch_size):
            if self.cursor >= len(self.dataset):
                break

            data = self.dataset[self.cursor]
            batch_images.append(data[0])
            batch_labels.append(data[1])
            self.cursor += 1
        return np.array(batch_images), np.array(batch_labels)


def softmax(x):
    x = np.clip(x, -1e10, 100)
    ex = np.exp(x)
    sum_ex = np.sum(ex,axis=1, keepdims=True)
    result = ex / sum_ex
    result = np.clip(result, 1e-10, 1e10)
    return result


if __name__ == '__main__':
    train_images = load_images(os.path.join("data", "mnist", "train-images.idx3-ubyte")) / 255
    train_labels = load_labels(os.path.join("data", "mnist", "train-labels.idx1-ubyte"))

    dev_images = load_images(os.path.join("data", "mnist", "t10k-images.idx3-ubyte")) / 255
    dev_labels = load_labels(os.path.join("data", "mnist", "t10k-labels.idx1-ubyte"))

    train_labels = make_onehot(train_labels, 10)

    train_images = train_images.reshape(60000, 784)
    dev_images = dev_images.reshape(-1, 784)

    batch_size = 20
    shuffle = False

    train_dataset = Dataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)

    dev_dataset = Dataset(dev_images, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle)

    w1 = np.random.normal(0, 1, size=(784, 1024))
    w2 = np.random.normal(0, 1, size=(1024, 10))
    b1 = np.random.normal(0, 1, size=(1, 1024))
    b2 = np.random.normal(0, 1, size=(1, 10))

    epoch = 200
    lr = 0.001

    for e in range(epoch):
        for batch_images, batch_labels in train_dataloader:
            H = batch_images @ w1 + b1
            pre = H @ w2 + b2
            p = softmax(pre)
            loss = -np.mean(batch_labels * np.log(p))

            G = (p - batch_labels) / batch_images.shape[0]

            delta_w2 = H.T @ G

            delta_H = G @ w2.T

            delta_w1 = batch_images.T @ delta_H

            delta_b2 = np.sum(G, axis=0, keepdims=True)
            delta_b1 = np.sum(delta_H, axis=0, keepdims=True)

            w1 -= delta_w1 * lr
            w2 -= delta_w2 * lr
            b1 -= delta_b1 * lr
            b2 -= delta_b2 * lr

        print(loss)

        right_num = 0
        for batch_images, batch_labels in dev_dataloader:
            H = batch_images @ w1 + b1
            pre = H @ w2 + b2

            pre_idx = np.argmax(pre, axis=-1)

            right_num += np.sum(pre_idx == batch_labels)

            # image = dev_images[idx: idx + 1]
            # label = dev_labels[idx]
            # pre = image @ w + b
            # pre_idx = int(np.argmax(pre, axis=1))
            # right_num += int(pre_idx == label)

        acc = right_num / len(dev_images)
        print(f"acc:{acc:.3f}")



