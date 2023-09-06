"""
添加dataset，batch_size概念。
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


if __name__ == '__main__':
    train_images = load_images(os.path.join("data", "mnist", "train-images.idx3-ubyte")) / 255
    train_labels = load_labels(os.path.join("data", "mnist", "train-labels.idx1-ubyte"))

    dev_images = load_images(os.path.join("data", "mnist", "t10k-images.idx3-ubyte")) / 255
    dev_labels = load_labels(os.path.join("data", "mnist", "t10k-labels.idx1-ubyte"))

    train_labels = make_onehot(train_labels, 10)

    train_images = train_images.reshape(60000, 784)
    dev_images = dev_images.reshape(-1, 784)

    batch_size = 10
    shuffle = False

    train_dataset = Dataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)

    w = np.random.normal(0, 1, size=(784, 10))
    b = np.random.normal(0, 1, size=(1, 10))

    epoch = 100
    lr = 0.01

    for e in range(epoch):
        for batch_images, batch_labels in train_dataloader:
            pass

        right_num = 0
        for idx in range(len(dev_images)):
            image = dev_images[idx: idx + 1]
            label = dev_labels[idx]

            pre = image @ w + b
            pre_idx = int(np.argmax(pre, axis=1))
            right_num += int(pre_idx == label)

        acc = right_num / len(dev_images)
        print(f"acc:{acc:.3f}")



