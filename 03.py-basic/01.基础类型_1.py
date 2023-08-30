# 整数，浮点数（小数，8，16，32，64），字符串，list，tuple，dict，set，bool

a = 10 + 2
print(type(a))

b = 10.2
print(type(b))

c = int(b)
print(c)


c = int(b + 0.5)  # 四舍五入 让一个浮点型 + 0.5
print(c)


s1 = "adfa"
print(s1)

s2 = "'fsda' "
s3 = """ dsafsa "fdasf" """

print(s2[0])

s4 = '1234567'
s4[0] = 'ss'  # TypeError: 'str' object does not support item assignment
print(s4[0:5:1])  # 1 作为取值的方向
print(s4[2:5:-1])
print(s4[1:5:2])

#####
f = '123稍等325432dsafasd445'  # 保留数字，如何做？

print(s4[-1:-3:-1])  # -1 作为取值的方向

list1 = [12, 2, 3, 5, "adf", 23.3, [123], ['ddd']]
print(list1[0: 3])  # 切片切出来的不会降维度
print(list1[-1])  # 取元素,索引降维度

list2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(list2[0])  # [1, 2, 3]  # 取元素,索引降维度
print(list2[0:1])  # [[1, 2, 3]]  # 切片切出来的不会降维度

t1 = (3, 4, 5, 6)
print(t1[0])
t1[3] = 55  # TypeError: 'tuple' object does not support item assignment

a = (1, 3)
b = (10)  # int
c = (2, )
d = 3, 4, 5, 9
print(type(d))  # tuple


