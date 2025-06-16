import pandas as pd
import numpy as np
import matplotlib.pyplot as plt          # For plotting graphs
import seaborn as sns                    # For advanced visualizations
from sklearn.model_selection import train_test_split  # For splitting training data
from sklearn.ensemble import RandomForestClassifier   # Our ML model
from sklearn.metrics import classification_report     # To evaluate model
from sklearn.preprocessing import LabelEncoder        # To convert text to numbers


train_df = pd.read_csv('train.csv')     # This file contains both features and labels
test_df = pd.read_csv('test.csv')       # This file contains only features
print(train_df.head())                  # Shows first 5 rows of the training data
print(train_df.info())                  # Shows column names, data types, missing values
print(train_df.isnull().sum())          # Counts missing values in each column
train_df.fillna(method='ffill', inplace=True)   # Forward fill missing values
test_df.fillna(method='ffill', inplace=True)
train_df.drop('Loan_ID', axis=1, inplace=True)  # Remove ID column from training
test_df.drop('Loan_ID', axis=1, inplace=True)   # Remove ID column from test
le = LabelEncoder()
train_df['Loan_Status'] = le.fit_transform(train_df['Loan_Status'])  
# Converts 'Y' to 1 and 'N' to 0
train_df = pd.get_dummies(train_df, drop_first=True)  # One-hot encode categorical columns
test_df = pd.get_dummies(test_df, drop_first=True)
# Get any columns in train that are missing in test
missing_cols = set(train_df.columns) - set(test_df.columns)
missing_cols.discard('Loan_Status')  # Remove target column from this list

# Add those columns with value 0 in test
for col in missing_cols:
    test_df[col] = 0

# Reorder test columns to match train (excluding target)
test_df = test_df[train_df.drop('Loan_Status', axis=1).columns]
X = train_df.drop('Loan_Status', axis=1)   # All columns except target
y = train_df['Loan_Status']                # Target column (0 or 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Splits the training data into 80% train and 20% validation
model = RandomForestClassifier(random_state=42)  # You can use other models too
model.fit(X_train, y_train)                      # Model learns patterns from the data
y_val_pred = model.predict(X_val)                # Predict on validation set
print(classification_report(y_val, y_val_pred))  # Print accuracy, precision, recall, etc.
y_test_pred = model.predict(test_df)             # Predict on final test set
