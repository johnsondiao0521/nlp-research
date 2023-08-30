

class Apple:
    s = 10  # 类对象
    b = 20

    def __init__(self):
        print("apple is creating")
        self.fun1()

    def fun1(self):
        print("hello!!!!")


class Param:
    lr = 0.001
    epoch = 20
    batch_size = 20


a = Apple()
print(a.s)
print(Apple.s)
print(Apple.b)

print(Param.lr)
print(Param.epoch)
print(Param.batch_size)