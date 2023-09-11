"""
@Desc:完善SGD的zero_grad，添加Adam、MSGD类
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
    # x = np.clip(x, -1e10, 100)
    max_x = np.max(x, axis=-1, keepdims=True)
    x = x - max_x
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    result = ex / sum_ex
    result = np.clip(result, 1e-10, 1e10)
    return result


def sigmoid(x):
    x = np.clip(x, -100, 1e10)
    result = 1 / (1 + np.exp(-x))
    return result


class Module:
    def __init__(self):
        self.info = "Module:\n"
        self.params = []

    def __repr__(self):
        return self.info
    

class Parameters:
    def __init__(self, weight):
        self.weight = weight
        self.grad = np.zeros_like(self.weight)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.x = None
        self.info = f"Linear({in_features}, {out_features})"
        self.W = Parameters(np.random.normal(0, 1, size=(in_features, out_features)))
        self.B = Parameters(np.zeros((1, out_features)))

        self.params.append(self.W)
        self.params.append(self.B)

    def forward(self, x):
        self.x = x # 存起来，方便backward的时候运用
        result = x @ self.W.weight + self.B.weight
        return result
    
    def backward(self, G):
        self.W.grad = self.x.T @ G
        self.B.grad = np.mean(G, axis=0, keepdims=True)

        self.W.weight -= lr * self.W.grad  # lr是全局变量，声明在main中
        self.B.weight -= lr * self.B.grad

        delta_x = G @ self.W.weight.T

        return delta_x  # 实际上返回的是对B矩阵的倒数


class Conv2D(Module):
    def __init__(self,in_channel,out_channel):
        super(Conv2D, self).__init__()
        self.info += f"     Conv2D({in_channel, out_channel})"
        self.W = Parameters(np.random.normal(0, 1, size=(in_channel, out_channel)))
        self.B = Parameters(np.zeros((1, out_channel)))

        self.params.append(self.W)
        self.params.append(self.B)

    def forward(self, x):
        result = x @ self.W.weight + self.B.weight

        self.x = x
        return result

    def backward(self, G):
        self.W.grad = self.x.T @ G
        self.B.grad = np.sum(G, axis=0)

        # self.W.weight -= lr * (self.W.grad)
        # self.B.weight -= lr * (self.B.grad)

        delta_x = G @ self.W.weight.T

        return delta_x


class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.weight -= self.lr * param.grad

    def zero_grad(self):
        pass


class MSGD:
    def __init__(self, parameters, lr=0.3, u=0.1):
        self.parameters = parameters
        self.lr = lr
        self.u = u

        for param in self.parameters:
            param.last_grad = 0

    def step(self):
        for param in self.parameters:
            param.weight = param.weight - self.lr * ((1 - self.u) * param.grad + self.u * param.last_grad)
            param.last_grad = param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0


class Adam():
    def __init__(self):
        pass


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.result = None
        self.info = f"Sigmoid"

    def forward(self, x):
        self.result = sigmoid(x)
        return self.result

    def backward(self, G):
        return G * self.result * (1 - self.result)


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.result = None
        self.info = f"Tanh"

    def forward(self, x):
        self.result = 2 * sigmoid(2 * x) - 1
        return self.result

    def backward(self, G):
        return G * (1 - self.result ** 2)


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.negative = None
        self.info = f"ReLU"

    def forward(self, x):
        self.negative = x < 0
        x[self.negative] = 0
        return x

    def backward(self, G):
        G[self.negative] = 0
        return G


class Dropout(Module):
    def __init__(self, rate=0.3):
        super(Dropout, self).__init__()
        self.negative = None
        self.rate = rate
        self.info += f"** Dropout({rate})"

    def forward(self, x):
        r = np.random.rand(*x.shape)
        self.negative = r < self.rate
        x[self.negative] = 0
        return x

    def backward(self, G):
        G[self.negative] = 0
        return G


class Softmax(Module):
    def __init__(self):
        super(Softmax, self).__init__()
        self.p = None
        self.info = f"Softmax"

    def forward(self, x):
        self.p = softmax(x)
        return self.p

    def backward(self, label):
        G = (self.p - label) / len(label)
        return G


class ModuleList:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, G):
        for layer in self.layers[::-1]:
            G = layer.backward(G)
        return G

    def __repr__(self):
        info = ""
        for layer in self.layers:
            info += layer.info
            info += "\n"

        return info


class Model:
    def __init__(self):
        self.label = None
        self.model_list = ModuleList(
            [
                Linear(784, 256),
                ReLU(),
                Dropout(0.1),
                Conv2D(256, 300),
                Tanh(),
                Dropout(0.2),
                Linear(300, 10),
                Softmax()
            ]
        )

    def forward(self, x, label=None):
        pre = self.model_list.forward(x)

        if label is not None:
            self.label = label
            loss = -np.mean(label * np.log(pre))
            return loss
        else:
            return np.argmax(pre, axis=-1)

    def backward(self):
        self.model_list.backward(self.label)

    def __repr__(self):
        return self.model_list.__repr__()

    def parameters(self):
        all_parameters = []
        for layer in self.model_list.layers:
            all_parameters.extend(layer.params)

        return all_parameters


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
    epoch = 100
    lr = 0.01

    train_dataset = Dataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle)

    dev_dataset = Dataset(dev_images, dev_labels)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle)

    model = Model()
    # opt = SGD(model.parameters(), lr=lr)
    opt = MSGD(model.parameters(), lr=lr)
    print(model)

    for e in range(epoch):
        for x, l in train_dataloader:
            loss = model.forward(x, label=l)
            model.backward()

            opt.step()
            opt.zero_grad()

        print(loss)

        right_num = 0
        for x, batch_labels in dev_dataloader:
            pre_idx = model.forward(x)
            right_num += np.sum(pre_idx == batch_labels)

        acc = right_num / len(dev_images)
        print(f"acc:{acc:.3f}")



