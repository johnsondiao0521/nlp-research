"""
@Desc:这个版本实现了实现3层linear + 非线性激活函数（sigmoid）。
后面的文件会基于这个文件做封装，一点点的开始。
为什么要从这个版本开始：
1. acc 80%以上，
2. 模型训练的整个逻辑都编写完了。
激活函数作用：
1. 增加非线性因子
2. 控制上下限
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


def sigmoid(x):
    x = np.clip(x, -100, 1e10)
    result = 1 / (1 + np.exp(-x))
    return result


if __name__ == '__main__':
    train_images = load_images(os.path.join("data", "mnist", "train-images.idx3-ubyte")) / 255
    train_labels = load_labels(os.path.join("data", "mnist", "train-labels.idx1-ubyte"))

    dev_images = load_images(os.path.join("data", "mnist", "t10k-images.idx3-ubyte")) / 255
    dev_labels = load_labels(os.path.join("data", "mnist", "t10k-labels.idx1-ubyte"))

    train_labels = make_onehot(train_labels, 10)

    train_images = train_images.reshape(60000, 784)
    dev_images = dev_images.reshape(-1, 784)

    batch_size = 50
    shuffle = False

    train_dataset = Dataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)

    dev_dataset = Dataset(dev_images, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle)

    w1 = np.random.normal(0, 1, size=(784, 256))
    w2 = np.random.normal(0, 1, size=(256, 300))
    w3 = np.random.normal(0, 1, size=(300, 10))

    b1 = np.zeros((1, 256))
    b2 = np.zeros((1, 300))
    b3 = np.zeros((1, 10))

    epoch = 100
    lr = 0.0001

    for e in range(epoch):
        for batch_images, batch_labels in train_dataloader:
            H1 = batch_images @ w1 + b1  # 第一层
            H1_S = sigmoid(H1)  # 第二层
            H2 = H1_S @ w2 + b2  # 第三层
            pre = H2 @ w3 + b3  # 第四层
            p = softmax(pre)
            loss = -np.mean(batch_labels * np.log(p))

            G4 = (p - batch_labels) / batch_images.shape[0]

            delta_w3 = H2.T @ G4
            G3 = delta_H2 = G4 @ w3.T

            delta_w2 = H1_S.T @ G3
            delta_H1_S = G3 @ w2.T

            G1 = delta_H1 = delta_H1_S * (H1_S * (1 - H1_S))
            delta_w1 = batch_images.T @ G1

            delta_b3 = np.mean(G4, axis=0, keepdims=True)
            delta_b2 = np.mean(G3, axis=0, keepdims=True)
            delta_b1 = np.mean(G1, axis=0, keepdims=True)

            w1 -= delta_w1 * lr
            w2 -= delta_w2 * lr
            w3 -= delta_w3 * lr
            b1 -= delta_b1 * lr
            b2 -= delta_b2 * lr
            b3 -= delta_b3 * lr

        print(loss)

        right_num = 0
        for batch_images, batch_labels in dev_dataloader:
            H1 = batch_images @ w1 + b1  # 第一层
            H1_S = sigmoid(H1)  # 第二层
            H2 = H1_S @ w2 + b2  # 第三层
            pre = H2 @ w3 + b3  # 第四层

            pre_idx = np.argmax(pre, axis=-1)

            right_num += np.sum(pre_idx == batch_labels)

        acc = right_num / len(dev_images)
        print(f"acc:{acc:.3f}")



