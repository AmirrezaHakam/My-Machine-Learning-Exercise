# Import necessary libraries
# For calculating distances between data points
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler  # For scaling features to a range
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For visualization (correct import)
from sklearn.cluster import AgglomerativeClustering  # For Agglomerative Clustering

# Load the dataset
pdf = pd.read_csv('cars_clus.csv')  # Read the CSV file
# print(pdf.head())  # Display the first few rows of the dataset (commented out)

# Print the shape of the dataset before cleaning
print(f"Shape of dataset before cleaning: {pdf.shape}")

# Convert specified columns to numeric types, coercing errors to NaN
pdf[['sales', 'resale', 'type', 'price', 'engine_s',
     'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
     'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
                               'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
                               'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values and reset the index
pdf = pdf.dropna()  # Remove any rows with NaN values
pdf = pdf.reset_index(drop=True)  # Reset the index of the DataFrame
# Print the shape after cleaning
print(f"Shape of dataset after cleaning: {pdf.shape}")
# print(pdf.head())  # Display the first few rows after cleaning (commented out)

# Prepare the feature set for clustering
featureset = pdf[['engine_s', 'horsepow', 'wheelbas',
                  'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Extract values from the feature set
x = featureset.values

# Scale features to a range [0, 1]
min_max_scaler = MinMaxScaler()  # Initialize the MinMaxScaler
feature_mtx = min_max_scaler.fit_transform(x)  # Fit and transform the data
# Display the first 5 rows of the scaled feature matrix
print(feature_mtx[0:5])

# Calculate the distance matrix using Euclidean distance
dist_matrix = euclidean_distances(feature_mtx, feature_mtx)

# Perform Agglomerative Clustering
agglom = AgglomerativeClustering(
    n_clusters=6, linkage='complete')  # Initialize clustering model
agglom.fit(dist_matrix)  # Fit the model to the distance matrix

# Get cluster labels
# Assign cluster labels to the original DataFrame
pdf['cluster_'] = agglom.labels_
print(pdf.head())  # Display the first few rows of the DataFrame with cluster labels
