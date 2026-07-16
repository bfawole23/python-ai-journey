#  writing csv file
# import csv brings in Python's built-in tool for handling CSV file
# csv.DictReader reads each row as a dictionary, using thje first line
#  
import csv 
with open("patients.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(f"{row['name']} is {row['age']} years old and has {row['condition']}")
        print(int(row['age']) + 5)

        if int(row['age']) >= 18:
            print("Adult")
        else: 
            print("Minor")


