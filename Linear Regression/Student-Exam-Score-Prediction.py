import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data=pd.read_csv("student_scores.csv")
print(data.head())
data.info()
print(data.isnull().sum())
print(data.describe())
print(os.getcwd())

X = data[["study_hours"]]
y = data["exam_score"]
print("Features:")
print(X.head())

print("Target:")
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict exam scores for testing data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Actual scores:", y_test.values)
print("Predicted scores:", y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
# Predict the score of a new student
new_student = pd.DataFrame({
    "study_hours": [7.5]
})

predicted_score = model.predict(new_student)

print("Study Hours:", new_student["study_hours"].iloc[0])
print("Predicted Exam Score:", predicted_score[0])

plt.scatter(
    data["study_hours"],
    data["exam_score"],
    label="Actual Scores"
)

plt.plot(
    data["study_hours"],
    model.predict(data[["study_hours"]]),
    color="red",
    label="Regression Line"
)

plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Student Exam Score Prediction")
plt.legend()
plt.show()

