# 匿名函数

y1 = lambda x: x+1
y2 = lambda x: x**2

print(y1(11))
print(y2(2))

str1 = "{'a': 1, 'b': 2}"
d = eval(str1)
print(d)

x = 1
y = eval("x+1")
print(y)

a = [1, 2, 3, 4]
b = 10
c = range(10)
d = "d1232"

for i in a:
    print(i)

# for i in b:
#     print(i)  TypeError: 'int' object is not iterable

for i in d:
    print(i)

print(dir(a))  # __new__ 前后有两个_是魔术方法，特定应用场景下触发的方法


#  __iter__  可迭代对象

list1 = [10, 21, 30, 4]
list2 = []
for i in list1:
    list2.append(i+1)
print(list2)

list3 = [i+1 for i in list1]
print(list3)  # 列表推导式等价于list2

