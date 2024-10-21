# Import necessary libraries
from sklearn import metrics  # For model evaluation
from sklearn.neighbors import KNeighborsClassifier  # For KNN classification
# For splitting data into train/test sets
from sklearn.model_selection import train_test_split
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting histograms
import pandas as pd  # For data manipulation
from sklearn import preprocessing  # For data scaling

# Load dataset
# Load the dataset into a pandas DataFrame
df = pd.read_csv('teleCust1000t.csv')
print("First 5 rows of the dataset:")
print(df.head(5))  # Display the first 5 rows of the dataset

# Check the distribution of the target variable (customer category 'custcat')
print("Value counts of 'custcat':")
# Count occurrences of each category in 'custcat'
print(df['custcat'].value_counts())

# Plot a histogram of 'income'
df.hist(column='income', bins=50)  # Plot income distribution using 50 bins
plt.show()  # Show the plot

# Display the column names of the dataset
print("Column names of the dataset:")
print(df.columns)

# Feature matrix (X): contains independent variables
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
       'employ', 'retire', 'gender', 'reside']].values
print("Feature matrix sample (first 5 rows):")
print(X[0:5])

# Target vector (Y): contains dependent variable ('custcat')
Y = df[['custcat']]
print("Target vector sample (first 5 rows):")
print(Y[0:5])

# Scale the features for standardization
scaler = preprocessing.StandardScaler().fit(
    X)  # Fit the scaler on the feature matrix
# Transform the features to standardized values
X = scaler.transform(X.astype(float))

# Split the dataset into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=4)

# Check the shapes of the training and testing sets
print(f"Train Set Shape: {x_train.shape}, {y_train.shape}")
print(f"Test Set Shape: {x_test.shape}, {y_test.shape}")

# K-Nearest Neighbors (KNN) model
K = 4  # Number of neighbors
neigh = KNeighborsClassifier(n_neighbors=K).fit(
    x_train, y_train)  # Train the KNN classifier

# Predict the customer category for the test set
y_hat = neigh.predict(x_test)

# Evaluate the model performance on the training and testing sets
print(
    f"Train Set Accuracy: {metrics.accuracy_score(y_train, neigh.predict(x_train))}")
print(f"Test Set Accuracy: {metrics.accuracy_score(y_test, y_hat)}")

# Determine the best value of K (1 to 9) by evaluating accuracy
Ks = 10  # Maximum number of neighbors to test
mean_acc = np.zeros((Ks-1))  # Initialize an array to store mean accuracies

# Loop through different values of K (1 to 9) to find the best K
for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(
        x_train, y_train)  # Train KNN with n neighbors
    y_hat = neigh.predict(x_test)  # Predict using the test set
    # Store the accuracy for each K
    mean_acc[n-1] = metrics.accuracy_score(y_test, y_hat)

# Print the accuracy for each K
print("Accuracy for different values of K:")
print(mean_acc)

# Optionally, you can plot the accuracy vs. K values to visualize the performance
plt.plot(range(1, Ks), mean_acc, 'g')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. K Neighbors')
plt.show()
