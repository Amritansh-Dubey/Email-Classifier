import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# 2. Preprocess data
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df['message'] = df['message'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# 4. Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train the model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Evaluate the model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🧩 Step 7 (Optional): Save and Test on New Messages
# ---------------------------------------------------
new_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "Limited time offer! Buy one get one free now.",
    "Hi Mom, just letting you know I arrived safely.",
]

# Convert the new messages using the same vectorizer
new_features = vectorizer.transform(new_messages)

# Make predictions
predictions = model.predict(new_features)

# Show results
for msg, pred in zip(new_messages, predictions):
    print(f"\nMessage: {msg}")
    print("Prediction:", "🚨 Spam" if pred == 1 else "✅ Not Spam")
