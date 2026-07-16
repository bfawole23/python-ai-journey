patient = {
    "name": "Amaka",
    "age": 34,
    "condition": "Hypertension"
}
print(patient)
print(patient["name"])
print(patient["age"])

doctor =  {
    "name": "Adaora",
    "specialty": "Dentist",
    "years of experience": 2
}

print(doctor)
print(doctor["name"])
print(doctor["specialty"])
print(doctor["years of experience"])

patients = [
    {"name": "Amaka", "age": 34, "condition": "Hypertension"},
    {"name": "David", "age": 25, "condition": "Hypertension"},
    {"name": "Bolu", "age": 12, "condition": "Hypertension"},
    {"name": "Dayo", "age": 62, "condition": "Hypertension"}
    
]

for patient in patients:
    print(f"{patient['name']} is {patient['age']} years old and has {patient['condition']}")

    if (patient['age']) >= 60:
        print("Needs monitoring")
    else:
        print("Routine care")
