def calculate_bmi(weight_kg, height_m):
    bmi = weight_kg / (height_m ** 2)
    bmi = round(bmi, 1)
    return bmi

result = calculate_bmi(70, 1.75)

print(result)
