# Import necessary libraries
import random  # For random operations
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation (not used here, but imported)
from sklearn.cluster import KMeans  # For K-Means clustering
# For generating synthetic data (blobs)
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt  # For plotting

# Set random seed for reproducibility
np.random.seed(0)

# Generate synthetic dataset with 4 centers (clusters)
x, y = make_blobs(n_samples=5000,  # Number of data points
                  centers=[[4, 4], [-2, -1], [2, -3],
                           [1, 1]],  # Centers of clusters
                  cluster_std=0.9)  # Standard deviation of clusters

# Visualize the generated data points in a scatter plot
# Plot each point (x, y) with a dot marker
plt.scatter(x[:, 0], x[:, 1], marker='.')
plt.title('Generated Data Points')  # Set title for the plot
plt.xlabel('X-axis')  # Label for the X-axis
plt.ylabel('Y-axis')  # Label for the Y-axis
plt.show()  # Display the scatter plot

# Perform K-Means clustering
k_means = KMeans(n_clusters=4,  # Number of clusters (centers)
                 init="k-means++",  # Method for initializing cluster centers
                 n_init=12)  # Number of different initializations to run
k_means.fit(x)  # Fit the K-Means model to the data

# Retrieve the labels (cluster assignments) for each data point
k_means_labels = k_means.labels_
print("Cluster Labels:")
print(k_means_labels)

# Retrieve the coordinates of the cluster centers
k_means_cluster_centers = k_means.cluster_centers_
print("Cluster Centers:")
print(k_means_cluster_centers)

# Visualizing the clustered data and centers
plt.scatter(x[:, 0], x[:, 1], c=k_means_labels, cmap='rainbow', marker='.')
plt.scatter(k_means_cluster_centers[:, 0], k_means_cluster_centers[:, 1],
            s=300, c='black', marker='X', label='Centroids')  # Plot cluster centers as 'X'
plt.title('Clustered Data with Centers')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()  # Add a legend for the centroids
plt.show()  # Display the plot
