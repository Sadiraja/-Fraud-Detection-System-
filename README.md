# -Fraud-Detection-System
How it Works

Data Preprocessing

Loads and standardizes the features from creditcard.csv.

Subsamples to 50,000 transactions for speed.

Handling Imbalance with SMOTE

Applies SMOTE to oversample the minority (fraud) class by 30%.

Training

Trains a custom RandomForest made of DecisionTree classifiers.

Evaluation

Outputs precision, recall, and F1-score on the test set.

Command-Line Interface

Lets you input new transactions manually to predict fraud/non-fraud.

