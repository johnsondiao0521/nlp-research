import os


def fun1(a, b, c=10):
    print(a, b, c)


def fun2(*args):  # 未命名变参
    print(type(args))  # tuple类型
    print(args)


def fun3(**kwargs):  # 命名变参
    print(type(kwargs))
    print(kwargs)


def fun4(a, b, c=10, *d, **e):
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)


if __name__ == '__main__':
    fun1(1, 2)
    print(1, 2, 3)

    fun2(1, 2)  # 未命名变参

    fun3(a=1, b=3)  # 命名变参

    fun4(1, 2, 3, 4, 5, 6, abc=22)
    fun4(1, 2, 3, 4, 5, cc=6, abc=22)