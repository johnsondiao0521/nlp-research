# tuple正确的排序
t = (3, 4, 56, 5, 2, 99)
a = list(t)  # 致命的
a.sort()
t2 = tuple(a)
print(t2)

list1 = [8, 4, 6, 2, 4, 55, 95]
list2 = sorted(list1)
print("list2:", list2)

list3 = list2[::-1]  # 倒序
print("list3:", list3)

list4 = sorted(list1, reverse=True)  # 倒序
print("list4:", list4)

list5 = [[5, 8, 7, -20], [3, 5], [3, 5, 4]]
list6 = sorted(list5)
print("list6:", list6)

y = lambda x: sum(x)

list6 = sorted(list5, key=y, reverse=True)
print("list6:", list6)
# list6: [[3, 5, 4], [3, 5], [5, 8, 7, -20]] sum从大到小

list6 = sorted(list5, key=lambda x: sum(x), reverse=True)  # 网上看到的形式是这样的
print("list6:", list6)

y2 = lambda x: x[-1]
list7 = sorted(list5, key=y2, reverse=True)
print("list7:", list7)
# list7: [[3, 5], [3, 5, 4], [5, 8, 7, -20]] 按照最后一个元素从大到小排序


list11 = [(5, 8, 7, -20), (3, 5), (3, 5, 4)]
aa = sorted(list11)
print("aa:", aa)
# aa: [(3, 5), (3, 5, 4), (5, 8, 7, -20)]

list22 = ((5, 8, 7, -20), (3, 5), (3, 5, 4))
bb = sorted(list22)
print("bb:", bb)
# bb: [(3, 5), (3, 5, 4), (5, 8, 7, -20)]

list33 = ([5, 8, 7, -20], [3, 5], [3, 5, 4])
cc = sorted(list33)
print("cc:", cc)
# cc: [[3, 5], [3, 5, 4], [5, 8, 7, -20]]
# 总结：里面的是什么就是什么，外面的会变为list

d1 = {'阿斯顿': 1, 'answer': -1, 'ca度答复': 0}
print("sorted(d1):", sorted(d1))
# sorted(d1): ['answer', 'ca度答复', '阿斯顿']  返回的是啥

d1 = {'阿斯顿': 1, 'answer': -1, 'ca度答复': 0}
print("sorted(d1):", sorted(d1, key=lambda x: d1[x]))
# sorted(d1): ['answer', 'ca度答复', '阿斯顿']  返回的是啥

d1 = {'阿斯顿': 1, 'answer': -1, 'ca度答复': 0}
print("sorted(d1):", sorted(d1.items()))
# sorted(d1): [('answer', -1), ('ca度答复', 0), ('阿斯顿', 1)]

d1 = {'阿斯顿': 1, 'answer': -1, 'ca度答复': 0}
print("sorted(d1):", sorted(d1.items(), key=lambda x:x[1]))
# sorted(d1): [('answer', -1), ('ca度答复', 0), ('阿斯顿', 1)]  返回的是啥

d2 = {"b": ("123", 11), "a": ("123", 10), "z": ("123", 99), "c": ("1234", 12)}
print("d2:", dict(sorted(d2.items(), key=lambda x: (len(x[1][0]), x[1][1]), reverse=True)))
print("d2:", dict(sorted(d2.items(), key=lambda x: (len(x[1][0]), -x[1][1]), reverse=True)))

# print(sorted(dict,key=lambda x : x.values))
