import pandas as pd

patients = [
    {"name": "Amaka", "age": 34, "condition": "Hypertension"},
    {"name": "David", "age": 25, "condition": "Diabetes"},
    {"name": "Chidi", "age": 19, "condition": "Asthma"},
    {"name": "Bolu", "age": 12, "condition": "ASthma"}
]

# pd.DataFrame(...) converts list to dictionaries
# df is a short of dataFrame
# df = pd.DataFrame(patients)

# check your data types before trusting yor data
df = pd.read_csv("patients.csv")
print(df)
print(df.head)
print(df.shape)
print(df["age"])

