import numpy as np


def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result


if __name__ == '__main__':
    dogs = np.array([[8.9, 10], [9, 11], [12.0, 12.2], [9.7, 12.2], [9.9, 13.1]], dtype=np.float64)
    cats = np.array([[4.9, 6], [5, 4], [2.0, 5.2], [4.7, 6.2], [4.0, 5.1]], dtype=np.float64)

    x = np.vstack((dogs, cats))

    label_2_index = {
        '狗': 0,
        '猫': 1
    }

    label = np.array([0 * len(dogs) + 1 * len(cats)]).reshape(-1, 1)

    b = 0
    weight = np.random.normal(0, 1, size=(2, 1))  # (10, 2) @ (2, 1) = (10, 1)

    lr = 0.001
    epoch = 10
    for e in range(epoch):
        out = x @ weight + b
        pre = sigmoid(out)
        loss = -np.mean((label * np.log(pre) + (1 - label) * np.log(1 - pre)))

        #  矩阵求导
        G = (pre - label) / weight.shape[0]

        delta_w = x.T @ G
        delta_b = np.sum(G)

        weight -= lr * delta_w
        b -= lr * delta_b

        print(loss)

    while True:
        f1 = float(input("请输入第一个特征:"))
        f2 = float(input("请输入第二个特征:"))

        p = np.array([f1, f2]).reshape(1, -1) @ weight + b

        p_sigmoid = sigmoid(p)

        if p_sigmoid[0][0] > 0.5:
            print("猫")
        else:
            print("狗")
