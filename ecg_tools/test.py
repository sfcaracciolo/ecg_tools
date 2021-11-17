import numpy as np

s = np.arange(6).reshape((6,1))
# s = np.asfortranarray(s)
print(s.flags)
it = np.nditer(s, flags=['external_loop'], order='F')
index = 0

for z in it:
# for z in s:
    print(z, index)
    index += 1