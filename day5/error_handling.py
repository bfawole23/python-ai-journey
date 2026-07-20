# i will be learning error handling with try/except
# 
import csv



try:
    with open("patients.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            age = int(row['age'])
            print(f"{row['name']} is {age} years old and has {row['condition']}")
except FileNotFoundError:
    print("That file doesn't exist. Check the filename and try again.")
except ValueError:
    print("Invalid data found in file")



