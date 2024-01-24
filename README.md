# Fraud Online Review Detection Using Machine Learning - Readme

## Overview

In the era of online reviews, the credibility of reviews has been compromised due to fraudulent activities. This project aims to address the issue of review fraud using machine learning methodologies and techniques, specifically focusing on natural language processing (NLP). The proposed approach involves training classifiers on Amazon's labeled dataset, utilizing the Verified Purchase label for supervised training.

## Methodology

### Data Source

The dataset used for training comes from Amazon, leveraging the Verified Purchase label as a ground truth for accurate reviews. This ensures that the models are trained on genuine user experiences.

### Classifiers

Three classifiers were employed for the supervised training:

1. **Multinomial Naive Bayes (MNB)**
2. **Support Vector Machine (SVM)**
3. **Logistic Regression (LR)**

### Vectorizers

Two distinct vectorizers were used for model tuning:

1. **CountVectorizer**
2. **TF-IDF Vectorizer**

## Model Performance

All trained models exhibited an accuracy rate of 80%, indicating their effectiveness in distinguishing between false and genuine reviews. The evaluation was based on the performance of the classifiers using two vectorization techniques. The CountVectorizer demonstrated a more significant improvement in model performance.

Among the three classifiers, Logistic Regression (LR) outperformed the others, achieving an accuracy rate of 85% and a recall rate of 92%. This suggests that LR is a robust choice for identifying fraudulent reviews in this context.

## Implementation

The LR classifier, chosen for its superior performance, was implemented and made accessible to the public. Users can input reviews, and the system provides a probability score indicating the likelihood of the entered reviews being genuine.

## Usage

To utilize the fraud detection system, users can input reviews into the LR classifier and receive a probability score. The higher the score, the more likely the review is genuine, and vice versa.

## Dependencies

Ensure that the following dependencies are installed to run the fraud detection system:

- Python 3.10.1
- Scikit-learn
- NLTK
- Pandas
- NumPy

