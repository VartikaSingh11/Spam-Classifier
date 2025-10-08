Spam Classifier
Project Overview

This project builds a machine learning model to classify SMS messages as spam or ham (not spam). The notebook loads the SMS spam dataset from spam.csv using pandas. It cleans the data by dropping unused columns and renaming the fields to Target (label) and Text (message). The Target column is label-encoded to numeric values (0 for ham, 1 for spam). The text messages are then preprocessed (lowercased, tokenized, stopwords removed, punctuation removed, and stemmed) before feature extraction. In total, the dataset contains 5572 messages, with the majority labeled as ham.

Features

Data Preprocessing: Drop irrelevant columns and rename headers. Encode the Target labels (ham/spam) to binary form.

Text Analysis Features: Compute message statistics such as number of characters, words, and sentences (added as new columns).

Text Cleaning: Convert text to lowercase, tokenize into words, remove English stopwords and punctuation, and apply stemming using NLTK.

Vectorization: Transform the cleaned text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

Model Training: Train classification algorithms on the TF-IDF features. The notebook demonstrates training Naive Bayes variants (Gaussian, Multinomial, Bernoulli), as well as other classifiers (SVM, Logistic Regression, Random Forest, etc.) and ensemble methods (voting, stacking).

Evaluation: Evaluate models using accuracy, confusion matrix, and precision scores. Key results (see Results below) are reported to compare classifier performance.

Serialization: The final trained model (e.g. MultinomialNB) and TF-IDF vectorizer are saved (pickled) for later use.

Technologies Used

Python 3 with Jupyter Notebook

Data processing: pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: scikit-learn (for TF-IDF, train/test split, classifiers, metrics)

Natural Language Processing: NLTK (tokenization, stopwords, stemming)

Additional Libraries: XGBoost (for gradient boosting classifier)



