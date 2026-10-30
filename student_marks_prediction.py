
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("student_data.csv")
print(df.head())

# Input and output
X = df[["Study_Hours"]]
y = df["Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Check model performance
print("Model R2 Score:", model.score(X_test, y_test))

# Predict marks
hours = pd.DataFrame({"Study_Hours": [7.5]})
prediction = model.predict(hours)

print("Study Hours:", 7.5)
print("Predicted Marks:", round(prediction[0], 2))

# Visualize results
plt.scatter(X["Study_Hours"], y, label="Actual Marks")

sorted_X = X.sort_values("Study_Hours")
plt.plot(
    sorted_X["Study_Hours"],
    model.predict(sorted_X),
    color="red",
    label="Prediction"
)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()
