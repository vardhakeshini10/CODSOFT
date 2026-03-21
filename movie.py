import pandas as pd

# Load dataset
train_data = pd.read_csv("train_data.txt", sep=":::", engine='python', header=None)

# Rename columns
train_data.columns = ["ID", "TITLE", "GENRE", "DESCRIPTION"]

print(train_data.head())

# Input and output
X = train_data["DESCRIPTION"]
y = train_data["GENRE"]

# Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom input
sample = ["A hero fights villains to save the world"]
sample_vec = vectorizer.transform(sample)

prediction = model.predict(sample_vec)
print("Predicted Genre:", prediction[0])