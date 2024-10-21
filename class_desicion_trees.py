# Import necessary libraries
from sklearn import metrics  # For model evaluation (accuracy score)
# For splitting data into training/testing sets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  # For encoding categorical features
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
# For creating the Decision Tree model
from sklearn.tree import DecisionTreeClassifier

# Load dataset
# Load the CSV file into a pandas DataFrame
my_data = pd.read_csv('drug200.csv')
print("First 5 rows of the dataset:")
print(my_data.head())  # Display the first 5 rows for inspection

# Feature matrix: X contains Age, Sex, BP, Cholesterol, and Na_to_K values
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Encode categorical variables using LabelEncoder

# Encoding 'Sex' (M/F) to numerical values (0/1)
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['M', 'F'])
# Replace the 'Sex' column with encoded values
X[:, 1] = le_sex.transform(X[:, 1])

# Encoding 'BP' (LOW, NORMAL, HIGH) to numerical values
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
# Replace the 'BP' column with encoded values
X[:, 2] = le_BP.transform(X[:, 2])

# Encoding 'Cholesterol' (NORMAL, HIGH) to numerical values
le_CH = preprocessing.LabelEncoder()
le_CH.fit(['NORMAL', 'HIGH'])
# Replace the 'Cholesterol' column with encoded values
X[:, 3] = le_CH.transform(X[:, 3])

# The encoded feature matrix X is now ready for model training
# Uncomment the following lines if you'd like to see a sample of the processed features
# print("Processed feature matrix (first 5 rows):")
# print(X[0:5])

# Target vector: Y contains the drug classification
Y = my_data[['Drug']]

# Split the dataset into training and testing sets (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=3)

# Create a Decision Tree classifier with the 'entropy' criterion and a maximum depth of 4
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)

# Train the model on the training data
drugTree.fit(x_train, y_train)

# Predict the drug type for the test data
predTree = drugTree.predict(x_test)

# Evaluate the accuracy of the model using the test data
accuracy = metrics.accuracy_score(y_test, predTree)
print(f"Decision Tree Accuracy: {accuracy}")
