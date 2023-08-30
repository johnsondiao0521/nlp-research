

class Apple:
    def __init__(self):  # 创建的时候触发
        print('init init')

    def fun1(self):
        pass

    def __repr__(self):  # print的时候触发
        return "hello apple"

    def __len__(self):  # 在len的时候触发
        return 10

    def __int__(self):
        return 99

    def __iter__(self):
        pass

    def __next__(self):  # 在next的时候触发
        pass


a = Apple()
print(a)

b = len(a)
print(b)

c = int(a)
print(c)

list1 = [10, 34, 43, 56, 786]
print(list1)
list1 = iter(list1)

print(next(list1))
print(next(list1))
print(next(list1))


print("hello")
print("world")

print(next(list1))
print(next(list1))
print(next(list1))  # 取完会报错 StopIteration

