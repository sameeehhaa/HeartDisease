import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("dataset.csv")

# -------------------------------
# Features and target
# -------------------------------
X = df.drop(columns=["target"])
y = df["target"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Predict and evaluate
# -------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.4f}")

# -------------------------------
# Save the model
# -------------------------------
joblib.dump(model, "heart_disease_model.pkl")

# -------------------------------
# Plot Actual vs Predicted (First 50)
# -------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(range(50), y_test[:50], color='blue', label='Actual', marker='o')
plt.scatter(range(50), y_pred[:50], color='orange', label='Predicted', marker='x')
plt.title("Actual vs Predicted - Heart Disease Risk (First 50 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Disease Presence (0 or 1)")
plt.yticks([0, 1])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")  # Save the plot as image
plt.show()
