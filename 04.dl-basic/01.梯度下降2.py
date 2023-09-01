# (x1-3)**2 + (x2+4)**2 =  0
# x1 = 3 , x2 = -4
import numpy as np

# label = np.array([3,-4])
#
# x = np.array([5,5])

label = 0

x1 = 1
x2 = -1

lr = 0.02

epoch = 1000

for e in range(epoch):
    predict = (x1-3)**2 + (x2+4) ** 2

    loss = (predict-label)**2

    delta_x1 = 2 * (predict-label) * 2 * (x1-3)
    delta_x2 = 2 * (predict-label) * 2 * (x2+4)

    x1 = x1 - lr*delta_x1
    x2 = x2 - lr*delta_x2

    if e % 100 == 0:
        print(x1,x2,loss)