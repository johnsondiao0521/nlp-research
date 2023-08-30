import os
from 调用包2 import a2


def fun1():
    print("hello1")


if __name__ == '__main__':

    # 主函数的作用域
    print(a2)
    # print(a3)
    # fun2() 要声明在前面


def fun2():
    print("hello2")
