# import xxx
# import xxx as xx
# from xxx import xxx
# from xxx import xxx, xxx
# from xxx import xxx as xxx
# from xxx import *
# from xxx.xxx import xxx as xxx

from 调用包2 import a2
a1 = 10
print(a2)

import 调用包2
print(调用包2.a2)

import 调用包2 as d
print(d.a2)

from 调用包2 import *
print(a2)
print(b2)

from import_page.func1 import a, b, c
print(a)
print(b)
print(c)