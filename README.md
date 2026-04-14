# Student Performance Prediction (Machine Learning Project)

## Overview
This project uses machine learning to predict whether a student will pass or fail based on demographic, behavioral, and academic features. The dataset comes from the UCI Machine Learning Repository and includes variables such as study time, absences, parental education, and previous grades.

## Dataset
- Source: UCI Machine Learning Repository  
- Dataset: Student Performance Dataset  
- File: data/student-mat.csv  

## Project Structure
data/        raw dataset  
outputs/     generated results, plots, and model outputs  
src/         source code (EDA, preprocessing, model training)  
Paper.pdf    final project report  

## Methods
The project follows a standard machine learning pipeline:
- Data loading and exploration (EDA)
- Data preprocessing (handling categorical and numeric features)
- Train/test split
- Model training and evaluation

The following models are implemented:
- Logistic Regression
- Decision Tree
- Random Forest

## Results
- Logistic Regression: 0.9873  
- Decision Tree: 1.0000 (likely overfitting)  
- Random Forest: 0.9873  

Feature importance analysis shows that previous grades (G1, G2, G3) are the most important predictors of student performance, followed by failures and absences.

## How to Run

1. Activate virtual environment:
source .venv/bin/activate

2. Run exploratory data analysis:
python3 src/eda.py --input data/student-mat.csv

3. Preprocess data:
python3 src/preprocess.py --input data/student-mat.csv

4. Train and evaluate models:
python3 src/train_models.py --train outputs/train_split.csv --test outputs/test_split.csv

## Author
Jinghan Fu
