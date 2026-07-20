from sklearn.linear_model import LogisticRegression
import numpy as np

# Training data: [age, health_score]
# x is the input data [age, health_score]
# y is the correct answer (0=low risk, 1=hgh risk)it is called labeled data

x = np.array([
    [25, 80],
    [30, 75],
    [45, 60],
    [50, 55],
    [65, 40],
    [70, 35],
    [22, 85],
    [60, 45]

])

# Labels: 0 = low risk, 1 = high risk
y = np.array([0, 0, 0, 1, 1, 1, 0, 1])

# model.fit(X, y) is the actual "learning" step — the model studies the patterns between X and y
model = LogisticRegression()
model.fit(x,y)

# Predict for a new, unseen patient: age 55, health_score 50
new_patient = np.array([[50, 50]])

# model.predict(...) then makes a guess on a brand new patient it's never seen before
prediction = model.predict(new_patient)
print(prediction)

young_healthy = np.array([[20, 90]])
old_unhealthy = np.array([[75, 30]])

print(model.predict(young_healthy))
print(model.predict(old_unhealthy))



