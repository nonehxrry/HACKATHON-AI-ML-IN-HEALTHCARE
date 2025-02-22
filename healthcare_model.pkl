import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load a sample healthcare dataset (e.g., Heart Disease dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
    "oldpeak", "slope", "ca", "thal", "target"
]
data = pd.read_csv(url, names=column_names, na_values="?")

# Drop rows with missing values
data = data.dropna()

# Prepare features and target
X = data.drop("target", axis=1)  # Features
y = data["target"].apply(lambda x: 1 if x > 0 else 0)  # Binary target: 0 = no disease, 1 = disease

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, "healthcare_model.pkl")
print("Model saved as healthcare_model.pkl")
