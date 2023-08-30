set1 = {1, 2, 3, 4, 5}
print(set1)

list1 = [12, 234, 54, 44, 1, 43, 2, 1]
list2 = list(set(list1))
print(list1)

# print(set1[0])  # 'set' object is not subscriptable

for i in set1:
    print(i)

# for i in range(len(set1)):
#     print(set1[i])  TypeError: 'set' object is not subscriptable
