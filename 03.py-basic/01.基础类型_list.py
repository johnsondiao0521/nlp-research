list1 = [1, 2, 3, 4]

a = list1  # 对
# b, c = list1  # too many values to unpack (expected 2)
# d, e, f = list1  # too many values to unpack (expected 3)
g, h, i, j = list1  # 对，解包

list2 = [1, 2, (3, 4, [5, 6])]

a, b, (c, d, (e, f)) = list2
a, b, c = list2
a, b, (c, d, e) = list2

print(a, b, c, d, e, f)

list3 = [(1, 2, "a"), (3, 4, "b"), (5, 6, "c"), (7, 8, "d")]

a, b, c = zip(*list3)
print(a)
print(b)
print(c)