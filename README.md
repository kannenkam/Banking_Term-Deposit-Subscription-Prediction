# Banking Term Deposit Subscription Prediction (Python)

## Overview
This project demonstrates a supervised machine learning use case in the Banking industry. 
The goal is to predict whether a bank customer will subscribe to a term deposit based on their demographic, financial, and campaign-related information.

This project is structured as a professional, modular Python codebase, suitable for demonstrating ML skills to prospective employers.

## Business Context
Banks conduct marketing campaigns to offer term deposits to potential customers. 
Targeting the right customers improves campaign success, reduces marketing costs, and enhances customer satisfaction.

Key benefits:
- Identify likely subscribers
- Optimize marketing spend
- Improve conversion rate

## Problem Statement
Predict whether a customer will subscribe to a term deposit (`y=1`) or not (`y=0`) using features such as:
- Age, job, marital status
- Account balance and loan status
- Previous campaign contact info (type, last contact duration)

## Approach
- Supervised machine learning (classification)
- Random Forest classifier as baseline
- Feature engineering:
  - Categorical encoding (one-hot / label encoding)
  - Numerical scaling
  - Derived features (optional)
- Model evaluation using Accuracy, Precision, Recall, F1-score

## Folder Structure
