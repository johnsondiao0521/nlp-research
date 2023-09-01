"""
sqrt(3) = ?
x**2 = 3, x=?
"""
epoch = 100000
label = 3
x = 5
lr = 0.00001

for e in range(epoch):
    pre = x ** 2
    loss = (pre - label) ** 2

    delta_x = 2 * (pre - label) * 2 *x

    x = x - lr * delta_x

    print(x, loss)
