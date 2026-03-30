import pandas as pd

# Load dataset
train_data = pd.read_csv("fraudTrain.csv")

print("Dataset Preview:")
print(train_data.head())

# Check fraud distribution
print("\nFraud vs Normal:")
print(train_data['is_fraud'].value_counts())

# ✅ Select only useful numerical columns
X = train_data[['amt', 'lat', 'long', 'city_pop', 'unix_time']]
y = train_data['is_fraud']

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

print("\nAccuracy:", accuracy_score(y_test, y_pred))