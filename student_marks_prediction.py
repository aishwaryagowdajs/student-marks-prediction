import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create dataset
df = pd.DataFrame({
    "Study_Hours": [2, 3, 4, 5, 6, 7, 8],
    "Marks": [30, 40, 50, 60, 70, 80, 90]
})

# Save dataset
df.to_csv("student_data.csv", index=False)

# Display dataset
print("Student Dataset:")
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
print("\nModel R2 Score:", round(model.score(X_test, y_test), 2))

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
plt.grid(True)
plt.show()