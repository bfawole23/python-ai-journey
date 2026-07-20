import numpy as np
import matplotlib.pyplot as plt


# np.random.seed(42) makes the random reproducable
np.random.seed(42)
#  .random — a sub-module inside NumPy specifically for generating random numbers
# .randint(...) — a function that generates random integers (whole numbers, no decimals)
# np.random.randint(1, 90, 200) generates 200 random "patient ages"
# everyone who runs this with seed 42 gets the exact same "random" data, which matters a lot for testing and sharing results
age = np.random.randint(1, 90, 200)

# (1, 90, 200) — three arguments passed into that function, each with a specific job:
# 1 — the lowest possible number (inclusive) — so 1 could actually appear
# 90 — the upper boundary (exclusive) — so it'll generate up to 89, never 90 itself. This is the exact same "stops before the end number" behavior as range() from Day 1 — same rule, different function.
# 200 — how many random numbers to generate, not a value range at all — this is the count

#  bins=10 groups ages into 10 ranges (buckets) instead of showing all 200 individually
plt.hist(age, bins=10, edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution of 200 Patients")
plt.savefig("age_distributon.png")
print("Histogram saved!")


sample = np.random.randint(1, 90, 200)
print(sample)
print(len(sample))
print(sample.min())
print(sample.max())





np.random.seed(42)
ages = np.random.randint(1, 90, 200)

plt.hist(ages, bins=5, edgecolor="black")
plt.title("5 Bins")
plt.savefig("bins_5.png")
plt.clf()  # clears the chart so the next one doesn't overlap

plt.hist(ages, bins=30, edgecolor="black")
plt.title("30 Bins")
plt.savefig("bins_30.png")