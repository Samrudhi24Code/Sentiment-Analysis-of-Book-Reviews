# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:24:58 2025

@author: Dell
"""

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
# Objective: To have a sample dataset with reviews and sentiments (positive/negative) for training our model
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
# Impact: Using pandas DataFrame to structure the data makes it easier to manipulate and preprocess
df = pd.DataFrame(data)

# Display the first few rows of the dataset to ensure it looks correct
print(df.head())  # This is just for verification and to see a snapshot of the dataset

# Visualize the distribution of sentiment labels (positive/negative)
# Objective: Understand the balance of sentiments in the dataset, as unbalanced datasets can lead to biased models
sns.countplot(x='sentiment', data=df)
plt.title('Distribution of Sentiments')
plt.show()  # Visualization helps us check if we have equal representation of both sentiments

# Data Preprocessing: Define a function to clean and preprocess the reviews
def preprocess_text(text):
    # Convert text to lowercase
    # Why: To standardize text and make sure that 'Book' and 'book' are treated the same
    text = text.lower()
    
    # Remove URLs from the text
    # Why: URLs are irrelevant for sentiment analysis and may introduce noise
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove any characters that are not alphabets (like punctuation)
    # Why: Punctuation marks don't contribute to the sentiment analysis and can be removed
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text into words
    # Why: Tokenization splits the text into individual words, which are necessary for analysis
    words = word_tokenize(text)
    
    # Remove stopwords (common words that don't add much value for sentiment analysis)
    # Why: Words like 'the', 'is', 'in' don't affect the sentiment of the sentence
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Join the words back into a single string and return it
    return ' '.join(words)

# Apply the preprocessing function to the reviews column
# Impact: Cleaned and processed text data will result in better performance by eliminating irrelevant data
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Split the data into features (X) and target labels (y)
# X: Features (cleaned text) and y: Labels (sentiments)
X = df['cleaned_review']
y = df['sentiment']

# Split the data into training and testing sets (80% for training, 20% for testing)
# Why: To evaluate how well the model generalizes to new, unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using Bag-of-Words model (CountVectorizer)
# Why: Machine learning algorithms can only work with numerical data, so we need to convert the text into a numerical format
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # Fit and transform on training data
X_test_vec = vectorizer.transform(X_test)  # Only transform the test data

# Initialize the Naive Bayes classifier
# Why: Naive Bayes is a simple and effective classifier for text data, especially when using the Bag-of-Words model
model = MultinomialNB()

# Train the Naive Bayes model on the training data
# Why: Training the model allows it to learn the relationship between text and sentiment
model.fit(X_train_vec, y_train)

# Use the trained model to predict sentiments on the test data
# Why: After training, we need to evaluate the model on unseen data (test set) to estimate its performance
y_pred = model.predict(X_test_vec)

# Evaluate the model's performance
# Print accuracy of the model
# Why: Accuracy tells us the percentage of correct predictions the model made
print('Accuracy:', accuracy_score(y_test, y_pred))

# Print confusion matrix to see the number of correct and incorrect predictions
# Why: A confusion matrix helps us see whether the model is confusing positive reviews with negative ones or vice versa
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Print classification report to see precision, recall, and F1-score for each class
# Why: Precision and recall provide deeper insights into the performance, especially when dealing with imbalanced datasets
print('Classification Report:\n', classification_report(y_test, y_pred))
