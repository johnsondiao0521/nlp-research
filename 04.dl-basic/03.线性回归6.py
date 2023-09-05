# y = kx + b
import random
import numpy as np

# ------------- 数据获取 ---------------
xs1 = np.array([i for i in range(2000, 2023)])

xs2 = np.array([random.randint(60, 150) for i in range(2000, 2023)])
ys = np.array(
    [5000, 8000, 12000, 15000, 14000, 18000, 20000, 25000, 26000, 32000, 40000, 42000, 46000, 50000, 51000, 53000,
     53000, 54000, 57000, 58000, 59000, 59900, 60000])

# -------------- 数据处理 ------------
# min-max 归一化 ：  （x-min） / ( max - min)
x1_min = min(xs1)
x1_max = max(xs1)

x2_min = min(xs2)
x2_max = max(xs2)

y_min = min(ys)
y_max = max(ys)

# xs1_normal = [ (i-x1_min)/(x1_max-x1_min)  for i in xs1]
xs1_normal = (xs1 - x1_min) / (x1_max - x1_min)
xs2_normal = (xs2 - x2_min) / (x2_max - x2_min)

xs_normal = np.dstack([xs1_normal, xs2_normal])[0]

ys_normal = (ys - y_min) / (y_max - y_min)
ys_normal = ys_normal.reshape(-1, 1)
# xs2_normal = [ (i-x2_min)/(x2_max-x2_min)  for i in xs2]
# ys_normal = [ (i-y_min)/(y_max-y_min)  for i in ys]

# -------------- 参数定义 ------------
k = np.array([1.0, 1.0]).reshape(2, 1)
b = 0
lr = 0.2
epoch = 100

# y = k1 * x1  + k2 * x2 + b

# -------------------------- 模型训练  ------------
for e in range(epoch):
    pre = xs_normal @ k + b

    loss = (pre - ys_normal) ** 2

    G = delta_C = (pre - ys_normal) / xs_normal.shape[0]

    delta_k = xs_normal.T @ G
    delta_b = np.sum(G)

    k -= lr * delta_k
    b -= lr * delta_b

    print(loss)

while True:
    input_x1 = int(input("请输入年份："))
    input_x1_normal = (input_x1 - x1_min) / (x1_max - x1_min)

    input_x2 = int(input("请输入大小："))
    input_x2_normal = (input_x2 - x2_min) / (x2_max - x2_min)

    # p = k*(input_x1_normal + input_x2_normal) + b
    p = np.array([input_x1_normal, input_x2_normal]) @ k + b

    pp = p * (y_max - y_min) + y_min
    print(f"房价为：{pp}")

#     for x1,x2,y in zip(xs1_normal,xs2_normal,ys_normal):
#         # -------------- 求预测值 （模型预测、模型推理） ------------
#         # predict = k * (x1+x2) + b
#
#         predict = k1 * x1 + k2 * x2  + b
#
#         # 求解loss 值
#         loss = (predict - y) ** 2
#
#         # 计算梯度值
#
#         delta_k1 = (predict - y) * x1
#         delta_k2 = (predict - y) * x2
#
#         delta_b =  predict - y
#
#         # 更新参数
#         k1 = k1 - delta_k1*lr
#         k2 = k2 - delta_k2*lr
#         b -= delta_b*lr
#     print(loss)
# # 模型上线 及 预测
