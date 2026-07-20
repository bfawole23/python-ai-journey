import pandas as pd
import matplotlib.pyplot as plt

patients = [
    {"name": "Amaka", "age": 34, "condition": "Hypertension"},
    {"name": "David", "age": 25, "condition": "Diabetes"},
    {"name": "Chidi", "age": 19, "condition": "Asthma"},
    {"name": "Bolu", "age": 12, "condition": "Asthma"}

]

df = pd.DataFrame(patients)

plt.bar(df["name"], df["age"])
plt.xlabel("Patient")
plt.ylabel("Age")
plt.title("Patient Age")
plt.savefig("patient_ages.png")
print("Chart saved!")