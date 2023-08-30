dict1 = {"a": 1, "b": 2, "c": 3}

for i in dict1:
    print(i)  # a b c

for i, j in dict1.items():
    dict1[i] = j + 1
print(dict1)  # {'a': 2, 'b': 3, 'c': 4}


dict2 = {i: j+1 for i, j in dict1.items()}
print(dict2)
# 如果一个东西是可迭代的对象就可以用list强转


class A:
    def __iter__(self):
        print("hello")
        return "hello"

    def __next__(self):
        return "hello"


a = A()
# print(list(a))  # TypeError: iter() returned non-iterator of type 'str'


# 如果一个东西是可迭代的对象就可以用list强转
# 1. 什么是生成器
# 2. 什么是迭代器
# 3. 什么是可迭代对象
class B:
    def __iter__(self):
        print("hello")
        return self

    def __next__(self):
        return "hello"


b = B()
print(list(b))