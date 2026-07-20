import numpy as np

# turns regular list into a Numpy array
ages = np.array([23, 45, 76, 44, 88])
print(ages)
print(type(ages))

print(ages + 5)
print(ages * 2)
print(ages / 2)


print(np.mean(ages))
print(np.max(ages))
print(np.min(ages))
print(np.median(ages))
print(np.std(ages))