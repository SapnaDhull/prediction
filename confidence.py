import pandas as pd
import joblib

# Load the new CSV file containing names
new_data_df = pd.read_csv('file/data.csv')  # Replace 'your_new_dataset.csv' with your actual file path

# Extract last names from the new dataset
new_data_df['Last_Name'] = new_data_df['member_name'].apply(lambda x: x.split()[-1] if ' ' in x else x)

# Load the saved classifiers
loaded_classifier_religion = joblib.load('naive_bayes_religion_classifier.joblib')
loaded_classifier_category = joblib.load('naive_bayes_category_classifier.joblib')

# Load the saved vectorizer
loaded_vectorizer = joblib.load('count_vectorizer.joblib')

# Prepare new data for classification
new_data_X = new_data_df['member_name']

# Convert last names into numerical features using the loaded vectorizer
new_data_X_vec = loaded_vectorizer.transform(new_data_X)

# Make probability predictions on the new data for both religion and category
new_data_prob_religion = loaded_classifier_religion.predict_proba(new_data_X_vec)
new_data_prob_category = loaded_classifier_category.predict_proba(new_data_X_vec)

# Display the predictions along with confidence levels
for full_name, prob_religion, prob_category in zip(new_data_df['member_name'], new_data_prob_religion, new_data_prob_category):
    pred_religion = loaded_classifier_religion.classes_[prob_religion.argmax()]
    pred_category = loaded_classifier_category.classes_[prob_category.argmax()]
    max_confidence_religion = prob_religion.max()
    max_confidence_category = prob_category.max()
    print(f"Full Name: {full_name}, Religion Prediction: {pred_religion} (Confidence: {max_confidence_religion}), Category Prediction: {pred_category} (Confidence: {max_confidence_category})")