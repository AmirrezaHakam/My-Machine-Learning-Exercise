# Import necessary libraries
# Metric to evaluate model performance
from sklearn.metrics import jaccard_score
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
# Splits dataset into train/test sets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  # For data preprocessing (scaling features)
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation

# Import matplotlib for potential visualizations (although not used here)
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs

# Load the dataset into a pandas DataFrame
# Assuming the CSV is in the same directory
chrun_df = pd.read_csv('ChurnData.csv')
print("First 5 rows of the dataset:")
print(chrun_df.head())  # Display the first 5 rows of the dataset

# Select specific columns that are relevant for model training
chrun_df = chrun_df[['tenure', 'age', 'address', 'income',
                     'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]

# Convert the 'churn' column to integer type, as it may be in a different format
chrun_df['churn'] = chrun_df['churn'].astype('int')
print("Dataset after selecting relevant columns and converting 'churn' to integer:")
print(chrun_df.head())

# Define feature matrix X and target vector Y
X = np.asanyarray(chrun_df[['tenure', 'age', 'address', 'income',
                            'ed', 'employ', 'equip', 'callcard', 'wireless']])
Y = np.asanyarray(chrun_df['churn'])

# Standardize the features (mean=0, variance=1) to improve model performance
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X.astype(float))

# Split the dataset into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4)

# Create and train the Logistic Regression model
# C is the inverse of regularization strength, and 'liblinear' is the solver
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train, y_train)

# Make predictions on the test set
y_hat = LR.predict(x_test)

# Evaluate the model using the Jaccard index for class '1' (churned customers)
jaccard = jaccard_score(y_test, y_hat, pos_label=1)
print(f"Jaccard Score for churned customers (class 1): {jaccard}")
