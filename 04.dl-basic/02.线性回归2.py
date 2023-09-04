# y = kx + b

# ------------- 数据获取 （上海）---------------
xs = [i for i in range(2000, 2023)]
ys = [5000, 8000, 12000, 15000, 14000, 18000, 20000, 25000, 26000, 32000, 40000, 42000, 46000, 50000, 51000, 53000,
      53000, 54000, 57000, 58000, 59000, 59900, 60000]

# -------------- 数据处理 ------------
# min-max归一化  (x - min) / (max - min)
x_min = min(xs)
x_max = max(xs)

y_min = min(ys)
y_max = max(ys)

xs_normal = [(i - x_min) / (x_max - x_min) for i in xs]
ys_normal = [(i - y_min) / (y_max - y_min) for i in ys]

# -------------- 参数定义 ------------
k1 = 1
k2 = 2
b = 0
lr = 0.2
epoch = 100

# -------------------------- 模型训练  ------------
for e in range(epoch):
    for x, y in zip(xs_normal, ys_normal):
        # -------------- 求预测值 （模型预测、模型推理） ------------
        predict = k1 * x + k2 * x + b

        # 求解loss 值
        loss = (predict - y) ** 2

        # 计算梯度值
        # delta_k = 2 * (predict - y) * x
        # delta_b = 2 * (predict - y) * 1

        delta_k1 = (predict - y) * x
        delta_k2 = (predict - y) * x
        delta_b = (predict - y) * 1
        # 更新参数
        k1 = k1 - delta_k1 * lr
        k2 = k2 - delta_k2 * lr
        b -= delta_b * lr
    print(loss)

# 模型上线 及 预测
while True:

    input_x = int(input("请输入年份："))
    input_x_normal = (input_x - x_min) / (x_max - x_min)
    p = k1 * input_x_normal + k2 * input_x_normal + b

    pp = p * (y_max - y_min) + y_min  # 反归一化
    print(f"{input_x}年的房价为：{pp}")
