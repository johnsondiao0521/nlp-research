import sys
sys.path.append('../03.py-basic')
from 调用包2 import a2

a = 10
b = 20
c = 30


def f(a):
    # from 调用包2 import a2
    print(a+a2)