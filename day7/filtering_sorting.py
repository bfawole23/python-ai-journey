import pandas as pd
df = pd.read_csv("patients.csv")

adults = df[df["age"] >= 18]

print(adults)

sorted_df = df.sort_values("age")
print(sorted_df)

sorted_desc = df.sort_values("age", ascending=False)
print(sorted_desc)

# i created df["age_group"], then i did df["age"].apply(...)

# Normal functions
# def get_age_group(age):
#    return "Adult" if age >= 18 else "Minor"

# df["age_group"] = df["age"].apply(lambda age: "Adult" if age >= 18 else "Minor")
# print(df)

df["is_teens"] = df["age"].apply(lambda age: True if age >= 13 and age >= 19 else False)
print(df)
