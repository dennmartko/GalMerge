#Sharing a multiprocessing.Array sharedctype's memory location with a numpy array goes as follows:

from ctypes import c_float
from multiprocessing import Array
import numpy as np

#dimensions for the matrix
dim = (4,2)

#The Array function from multiprocessing can only allocate 1D arrays
a_mp = Array(c_float, dim[0]*dim[1]) #we set the datatype to c_float

print(f"Multiprocessing array: {a_mp[:]}")

#Create a numpy array at the same memory location
a_np = np.frombuffer(a_mp.get_obj(), dtype=c_float)

print(f"Numpy array: {a_np}")

#reshape the numpy array making it 2D (still sharing same memory)
b_np = np.reshape(a_np, dim)

print(f"Reshaped numpy array: {b_np}")

#Edit b_np
b_np[0] += 1

print(f"b_np has changed: {b_np}")

#a_np has also changed
print(f"a_np has also changed: {a_np}")

#The multiprocessing array has also changed!
print(f"a_mp has also changed: {a_np[:]}")