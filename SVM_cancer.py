# Import necessary libraries
from sklearn.metrics import f1_score  # For calculating the F1 score
from sklearn import svm  # For Support Vector Machine classifier
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.model_selection import train_test_split  # For splitting the dataset

# Load the dataset
# Read the CSV file containing cell samples
cell_df = pd.read_csv('cell_samples.csv')
print(cell_df.head())  # Display the first few rows of the dataset

# Check data types of the DataFrame
print(cell_df.dtypes)

# Clean the 'BareNuc' column by converting to numeric and dropping NaN values
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype(
    'int')  # Convert 'BareNuc' to integer type

# Check data types again to confirm changes
print(cell_df.dtypes)

# Define features and target variable
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh',
                      'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asanyarray(feature_df)  # Convert features to a NumPy array

# Convert target variable to integer type
cell_df['Class'] = cell_df['Class'].astype('int')
Y = np.asanyarray(cell_df['Class'])  # Convert target variable to a NumPy array

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4)  # 80% training, 20% testing

# Create and train the SVM classifier
clf = svm.SVC(kernel="rbf")  # Initialize SVM with RBF kernel
clf.fit(x_train, y_train)  # Fit the model on the training data

# Make predictions on the test set
y_hat = clf.predict(x_test)

# Evaluate the model using F1 score
# Calculate F1 score for class '4'
print(f"F1 Score = {f1_score(y_test, y_hat, pos_label=4)}")
print("Actual labels:", y_test)  # Print actual labels from the test set
print("Predicted labels:", y_hat)  # Print predicted labels
