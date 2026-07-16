patients = ["Amaka", "David", "Chidi"]

print(patients)
print(patients[0])
print(patients[1])
print(len(patients))

patients.append("Ngozi")
print(patients)

patients.remove("David")
print(patients)

for patient in patients:
    print(f"Patient: {patient}")

ages = [23, 45, 76, 44, 13, 12, 88]

for age in ages:
    if age >= 18:
        print(f"{age}: Adult")
    else:
        print(f"{age}: Minor")
    