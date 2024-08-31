# Credit Card Fraud Detection
## Overview
This project focuses on detecting fraudulent credit card transactions using advanced machine learning techniques. The goal is to create a reliable model that can differentiate between legitimate and fraudulent transactions, helping to minimize financial losses and improve security for users and financial institutions.

The project includes training a classification model, deploying it on a web server, and providing a user interface where users can input transaction data to get predictions.

## Technologies Used
Python: Core programming language for the project.
Pandas: Data manipulation and analysis library used for handling and preprocessing the dataset.
Scikit-learn: Machine learning library used for model building and evaluation.
Flask: Lightweight web framework for deploying the model and creating a simple user interface.
NumPy: Library for numerical computing, used for handling arrays and numerical data.
Jupyter Notebook: For development, data exploration, and initial model training.
## Dataset
The dataset used in this project is the Kaggle Credit Card Fraud Detection Dataset. It contains transactions made by European cardholders in September 2013, with a total of 284,807 transactions. Out of these, 492 are fraudulent, making the dataset highly imbalanced. Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Dataset Features:
Time: The number of seconds elapsed between this transaction and the first transaction in the dataset.
V1-V28: The result of a PCA transformation applied to the original features, which have been anonymized.
Amount: The transaction amount.
Class: The target variable (0 for legitimate transactions, 1 for fraudulent transactions).
Modeling
Problem Statement:
Given the imbalanced nature of the dataset, the main challenge is to accurately identify the minority class (fraudulent transactions) while minimizing false positives.

## Steps:
### Data Preprocessing:

Handling missing values (if any).
Scaling features using StandardScaler.
Splitting the dataset into training and testing sets.

### Model Training:
Various models were evaluated, including Logistic Regression, Random Forest, and Gradient Boosting.
The final model chosen was based on its performance in terms of precision, recall, and the F1-score.

### Model Evaluation:
The model was evaluated using metrics like confusion matrix, accuracy, precision, recall, and the F1-score.
The model was fine-tuned to handle the imbalanced nature of the dataset.

## Web Application
The model is deployed as a web application using Flask. Users can input transaction data into a web form, and the application will predict whether the transaction is legitimate or fraudulent.

## Conclusion
This project demonstrates the process of building a machine learning model to detect fraudulent credit card transactions and deploying it as a web application. The project covers data preprocessing, model training, evaluation, and web development, providing a comprehensive learning experience in handling real-world problems.
