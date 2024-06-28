import pandas as pd
import joblib

# Load the new CSV file containing names
new_data_df = pd.read_csv('file/Dharmendra_followers_name.csv')  # Replace 'your_new_dataset.csv' with your actual file path

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

# Make predictions on the new data for both religion and category
new_data_pred_religion = loaded_classifier_religion.predict(new_data_X_vec)
new_data_pred_category = loaded_classifier_category.predict(new_data_X_vec)

# Display the predictions
for full_name, pred_religion, pred_category in zip(new_data_df['member_name'], new_data_pred_religion, new_data_pred_category):
    print(f"Full Name: {full_name}, Religion Prediction: {pred_religion}, Category Prediction: {pred_category}")
