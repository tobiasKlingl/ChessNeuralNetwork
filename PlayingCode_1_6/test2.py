import numpy as np
from numba.types import string, int32, float32, ListType
from numba import typeof, typed, njit
from numba.typed import List
from numba.experimental import jitclass

spec1 = [
    ('name', string),
]
@jitclass(spec1)
class Class1(object):
    def __init__(self, name):
        self.name = name


spec2 = [
    ('name', string),
]
@jitclass(spec2)
class Class2(object):
    def __init__(self, name):
        self.name = name


specCombined = [
    ('name',   string),
    ('class1', Class1.class_type.instance_type),
    ('class2', Class2.class_type.instance_type),
]
@jitclass(specCombined)
class Combined(object):
    def __init__(self, name, class1, class2):
        self.name = name
        self.class1 = class1
        self.class2 = class2


        
class1 = Class1("name1")
class2 = Class2("name2")

print("class1.name =", class1.name)
print("class1.name =", class2.name)

combined = Combined("combined", class1, class2)

print("combined.name =", combined.name)
print("combined.class1.name =", combined.class1.name)
print("combined.class2.name =", combined.class2.name)

print("")

combined.class1.name = "new1"
print("combined.class1.name =", combined.class1.name)
print("class1.name =", class1.name)
