# Importing necessary libraries
from sklearn.preprocessing import StandardScaler  # For standardizing features
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For visualization (import correction)
from sklearn.cluster import KMeans  # For K-Means clustering
# For generating synthetic data (not used in this case)
from sklearn.datasets import make_blobs

# Load the dataset
cust_df = pd.read_csv('Cust_Segmentation.csv')  # Load the CSV file
print(cust_df.head())  # Display the first few rows of the dataset

# Dropping the 'Address' column as it's not needed for clustering
df = cust_df.drop("Address", axis=1)  # Removing the Address column
print(df.head())  # Display the first few rows of the dataset after column removal

# Preparing the feature set for clustering
x = df.values[:, 1:]  # Exclude the first column (customer ID)
x = np.nan_to_num(x)  # Replace NaN values with zero or finite numbers
# Standardize features to have zero mean and unit variance
clus_dataset = StandardScaler().fit_transform(x)
print(clus_dataset)  # Show the transformed dataset

# Setting the number of clusters
cluster_num = 3  # Define the number of clusters to form

# Initializing and fitting the K-Means model
k_means = KMeans(init='k-means++',  # K-Means++ initialization method
                 n_init=12,  # Number of different initial centroid seeds
                 n_clusters=cluster_num)  # Set the number of clusters to 3
k_means.fit(clus_dataset)  # Fit the K-Means model to the standardized dataset

# Retrieve cluster labels
labels = k_means.labels_  # Labels indicate which cluster each sample belongs to
print(labels)  # Display the assigned cluster for each data point

# Adding the cluster labels to the original dataset
df["Clus_km"] = labels  # Create a new column with the cluster labels
print(df.head(5))  # Display the first 5 rows with the cluster labels

# Grouping the dataset by the cluster labels and calculating the mean of each cluster
# Group by cluster labels and calculate the mean of each group
cluster_summary = df.groupby('Clus_km').mean()
print(cluster_summary)  # Display the summary of each cluster
