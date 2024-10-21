# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.cluster import DBSCAN  # For clustering
import matplotlib.pyplot as plt  # For plotting
import cartopy.crs as ccrs  # For map projections
import cartopy.feature as cfeature  # For map features
from pylab import rcParams  # For adjusting plot parameters

# Load the dataset
pdf = pd.read_csv('weather-stations20140101-20141231.csv')

# Adjust the figure size for the plot
rcParams['figure.figsize'] = (14, 10)

# Define latitude and longitude limits for the map
llon, ulon = -140, -50  # Longitude limits
llat, ulat = 40, 65  # Latitude limits

# Filter data based on defined latitude and longitude limits
pdf = pdf[(pdf['Long'] > llon) & (pdf['Long'] < ulon) &
          (pdf['Lat'] > llat) & (pdf['Lat'] < ulat)]

# Create a figure with Mercator projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Mercator()})

# Set the extent of the map (longitude and latitude boundaries)
ax.set_extent([llon, ulon, llat, ulat], crs=ccrs.PlateCarree())

# Add map features: coastlines, borders, and gridlines
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True)

# Optionally fill the continents with color
ax.add_feature(cfeature.LAND, color='white', alpha=0.3)

# Add shaded relief for a better visual appearance
ax.stock_img()

# Convert longitude and latitude to map coordinates and add them to the DataFrame
pdf['xm'] = pdf['Long'].tolist()
pdf['ym'] = pdf['Lat'].tolist()

# Plot each weather station on the map
for index, row in pdf.iterrows():
    ax.plot(row.xm, row.ym, marker='o', color='red', markersize=5,
            alpha=0.75, transform=ccrs.PlateCarree())

# Show the map plot
plt.show()

# Clean the data by removing rows with null 'Tm' values and reset the index
pdf = pdf[pd.notnull(pdf['Tm'])]
pdf = pdf.reset_index(drop=True)
print(pdf.head(5))  # Display the first few rows of the cleaned DataFrame

# Prepare the dataset for clustering
clus_dataset = pdf[['xm', 'ym']]
clus_dataset = np.nan_to_num(clus_dataset)  # Replace NaN values with 0
clus_dataset = StandardScaler().fit_transform(
    clus_dataset)  # Standardize features

# Perform DBSCAN clustering
db = DBSCAN(eps=0.15, min_samples=10).fit(clus_dataset)  # Fit the DBSCAN model
# Create a mask for core samples
core_sample_mask = np.zeros_like(db.labels_, dtype=bool)
# Set core samples in the mask
core_sample_mask[db.core_sample_indices_] = True
labels = db.labels_  # Get cluster labels
pdf['Clus_Db'] = labels  # Add cluster labels to the DataFrame

# Calculate the number of unique clusters
real_cluster_num = len(set(labels)) - \
    (1 if -1 in labels else 0)  # Exclude noise
cluster_num = len(set(labels))  # Total unique clusters
# Display the first few rows with cluster labels
print(pdf[["Stn_Name", "Tx", "Tm", "Clus_Db"]].head(5))
