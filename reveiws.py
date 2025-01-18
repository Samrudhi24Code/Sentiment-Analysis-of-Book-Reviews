# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:24:58 2025

@author: Dell
"""

# Business Objective:
# Develop a sentiment analysis model to classify book reviews into positive or negative categories.
# This will help businesses understand customer sentiment and improve their product offerings.



# Importing the required libraries
import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download necessary NLTK resources for tokenization and stopwords
nltk.download('punkt')  # Needed for tokenizing the text into words
nltk.download('stopwords')  # Needed to remove common stopwords that don't add value to sentiment analysis

# Sample dataset of book reviews
data = {
    'review': [
        'I loved this book! The characters were great and the plot was interesting.',
        'Horrible book, I didn\'t like the story at all.',
        'This was a good read. I enjoyed the pacing and the character development.',
        'The book was boring and didn\'t capture my attention.',
        'An amazing book! The writing style was wonderful.',
        'Not worth the time, the plot was predictable.',
        'I really enjoyed this book, the plot twists were amazing.',
        'It was an average read, nothing special about it.',
        'The book was very slow, I couldn\'t finish it.',
        'Excellent book! Highly recommended.'
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive']
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Visualize the distribution of sentiment labels
sns.countplot(x='sentiment', data=df)
plt.title('Distribution of Sentiments')
plt.show()

# Data Preprocessing: Define a function to clean and preprocess the reviews
def preprocess_text(text):
    text = text.lower()  # Standardize text
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    words = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)  # Join the words back into a single string

# Apply the preprocessing function to the reviews column
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Split the data into features (X) and target labels (y)
X = df['cleaned_review']
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using Bag-of-Words model
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the Naive Bayes model
model.fit(X_train_vec, y_train)

# Use the trained model to predict sentiments on the test data
y_pred = model.predict(X_test_vec)

# Evaluate the model's performance

# 1. Minimize False Positives and False Negatives
# Why: To ensure reviews are correctly classified, reducing errors in sentiment detection
# Print confusion matrix
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# 2. Maximize Accuracy
# Why: Ensure the model predicts the sentiments correctly as much as possible
print('Accuracy:', accuracy_score(y_test, y_pred))

# 3. Maximize F1-Score
# Why: Balance precision and recall, especially in case of imbalanced datasets
print('Classification Report:\n', classification_report(y_test, y_pred))
