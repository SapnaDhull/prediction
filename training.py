import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

# Load the CSV file containing names, gender, and category labels
df = pd.read_csv('file/Caste-Religion-Analysis.csv')  # Replace 'your_dataset.csv' with your actual file path

# Separate features (names) and labels (gender, religion, and category)
X = df['Name']
y_religion = df['Religion']
y_category = df['Category']

# Convert names into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_religion_train, y_religion_test, y_category_train, y_category_test = train_test_split(
    X_vec, y_religion, y_category, test_size=0.2, random_state=42
)

# Train a Naive Bayes classifier for religion prediction
classifier_religion = MultinomialNB()
classifier_religion.fit(X_train, y_religion_train)

# Train a Naive Bayes classifier for category prediction
classifier_category = MultinomialNB()
classifier_category.fit(X_train, y_category_train)

# Save the trained classifiers
joblib.dump(classifier_religion, 'naive_bayes_religion_classifier.joblib')
joblib.dump(classifier_category, 'naive_bayes_category_classifier.joblib')

# Save the vectorizer
joblib.dump(vectorizer, 'count_vectorizer.joblib')

# Load the saved classifiers
loaded_classifier_religion = joblib.load('naive_bayes_religion_classifier.joblib')
loaded_classifier_category = joblib.load('naive_bayes_category_classifier.joblib')

# Load the saved vectorizer
loaded_vectorizer = joblib.load('count_vectorizer.joblib')

# Example: Make predictions on the original data for both religion and category
original_data_pred_religion = loaded_classifier_religion.predict(X_vec)
original_data_pred_category = loaded_classifier_category.predict(X_vec)

# Display the predictions
for name, pred_religion, pred_category in zip(X, original_data_pred_religion, original_data_pred_category):
    print(f"Name: {name}, Religion Prediction: {pred_religion}, Category Prediction: {pred_category}")
