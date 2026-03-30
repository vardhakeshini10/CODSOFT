import pandas as pd

data = pd.read_csv("Churn_Modelling.csv")

print("Dataset Preview:")
print(data.head())

data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

data = pd.get_dummies(data, drop_first=True)

X = data.drop("Exited", axis=1)
y = data["Exited"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Prediction (0 = No Churn, 1 = Churn):", prediction[0])