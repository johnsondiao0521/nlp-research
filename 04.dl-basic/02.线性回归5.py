import random
import numpy as np

# y = kx + b # 矩阵版本

# ------------- 数据获取 （上海）---------------
xs1 = np.array([i for i in range(2000, 2023)])
xs2 = np.array([random.randint(60, 150) for i in range(2000, 2023)])
ys = np.array(
    [5000, 8000, 12000, 15000, 14000, 18000, 20000, 25000, 26000, 32000, 40000, 42000, 46000, 50000, 51000, 53000,
     53000, 54000, 57000, 58000, 59000, 59900, 60000])

# -------------- 数据处理 ------------
# min-max归一化  (x - min) / (max - min)
x1_min = min(xs1)
x2_min = min(xs2)
x1_max = max(xs1)
x2_max = max(xs2)

y_min = min(ys)
y_max = max(ys)

xs1_normal = (xs1 - x1_min) / (x1_max - x1_min)
xs2_normal = (xs2 - x2_min) / (x2_max - x2_min)
x_normal = np.dstack([xs1_normal, xs2_normal])[0]

ys_normal = (ys - y_min) / (y_max - y_min)
ys_normal = ys_normal.reshape(-1, 1)

# -------------- 参数定义 ------------
k = np.array([1.0, 1.0]).reshape(2, 1)
b = 0
lr = 0.01
epoch = 100

# -------------------------- 模型训练  ------------
for e in range(epoch):
    for x1, x2, y in zip(xs1_normal, xs2_normal, ys_normal):
        # -------------- 求预测值 （模型预测、模型推理） ------------
        predict = x_normal @ k + b

        # 求解loss 值
        loss = (predict - y) ** 2

        G = delta_C = predict - ys_normal
        # 计算梯度值

        delta_k = xs2_normal.T @ G
        delta_b = (predict - y) * 1
        # 更新参数

        b -= delta_b * lr
    print(loss)

# 模型上线 及 预测
while True:
    input_x1 = int(input("请输入年份："))
    input_x1_normal = (input_x1 - x1_min) / (x1_max - x1_min)

    input_x2 = int(input("请输入大小："))
    input_x2_normal = (input_x2 - x2_min) / (x2_max - x2_min)

    # p = k*(input_x1_normal + input_x2_normal) + b
    p = np.array([input_x1_normal, input_x2_normal]) @ k + b

    pp = p * (y_max - y_min) + y_min
    print(f"房价为：{pp}")
