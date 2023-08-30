list1 = [1, 3, 5, 5, 7]
list2 = [5, 6, 7]
list3 = [6, 7, 8, 4, 5]
t2 = (3, 4, 56, 2, 4, 65)

# list(t2).sort()

a = list(t2)
a.sort()

t3 = tuple(a)

print(t2)
print(t3)
# for i in range(5):
#     print(list1[i],list2[i])

for i, j, k in zip(list1, list2, list3):
    print(i, j, k)
