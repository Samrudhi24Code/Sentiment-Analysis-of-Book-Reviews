# Sentiment-Analysis-of-Book-Reviews
# Sentiment Analysis of Book Reviews

This project aims to develop a sentiment analysis system using Natural Language Processing (NLP) techniques to classify book reviews as positive or negative. The system processes raw text reviews, cleans and preprocesses them, and uses a Naive Bayes classifier for sentiment prediction.

## Objectives

- Implement sentiment analysis using NLP techniques.
- Clean and preprocess text data to prepare it for machine learning.
- Use the Bag-of-Words model to transform text data into numerical features.
- Train a Naive Bayes classifier to predict the sentiment of book reviews.
- Evaluate the model's performance using accuracy, confusion matrix, and classification report.

## Project Structure

- **data**: Contains sample book reviews and their associated sentiments (positive/negative).
- **code**: Python code that performs text preprocessing, feature extraction, model training, and evaluation.
- **output**: The printed results including accuracy, confusion matrix, and classification report.

## Technologies Used

- **Python**: The primary programming language.
- **Libraries**: 
  - `pandas`: Data manipulation.
  - `nltk`: For text preprocessing and tokenization.
  - `seaborn`, `matplotlib`: For data visualization.
  - `sklearn`: For machine learning tasks (e.g., Naive Bayes, CountVectorizer).

## Requirements

To run this project, you need to have Python 3.x installed on your machine. You will also need to install the following libraries:

- `pandas`
- `nltk`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using `pip`:

```bash
pip install pandas nltk seaborn matplotlib scikit-learn
