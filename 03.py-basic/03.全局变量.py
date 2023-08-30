import os
b = 20


def read_data():
    global b
    print(b)
    b += 1
    print(b)


read_data()
print(b)
