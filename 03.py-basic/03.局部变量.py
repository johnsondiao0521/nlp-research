def read_data(b):
    b += 1
    print(b)


b = 10
read_data(b)

print(b)
# ===================================================
list1 = [1, 2, 3, 4, 5]
for i in list1:
    print("hello", i)

print(i)

# ===================================================

a = 10
if a == 10:
    bb = 20
else:
    cc = 30
print(bb)
# print(cc) 报错：NameError: name 'cc' is not defined
# ===================================================

list2 = [13, 3, 4, 5]
for i in list2:
    if i % 2 == 0:
        a = 10
    else:
        b += 1
print(a)
print(b)

# ===================================================