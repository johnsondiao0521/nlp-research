d = {"a": 0, "b": 1, "c": 2}

k = d.keys()
print(k)
k = list(d.keys())
print(k)

v = d.values()
print(v)
v = list(d.values())
print(v)

print(d.get("a", "没有对应的key"))
print(d.get("aa", "没有对应的key"))

d["d"] = 33
print(d)
d.pop("d")
print(d)

d["d"] = 23
print(d)
del d["d"]
print(d)

list1 = [
    {"学号": 123, "姓名": "钉钉", "set": "男"},
    {"学号": 234, "姓名": "QQ", "set": "男"},
    {"学号": 567, "姓名": "微信", "set": "女"}
]

# zip() 函数用于将多个可迭代对象（比如列表、元组或字符串）打包成一个个元组，然后返回一个可迭代的 zip 对象。
# zip(*) 则是将 zip 对象进行解压缩和转置的方法。
# 它将 zip 对象中的元组按照索引位置重新组织，得到原始数据的转置形式。
nums, name, sex = zip(*[i.values() for i in list1])
print(nums)
print(name)
print(sex)