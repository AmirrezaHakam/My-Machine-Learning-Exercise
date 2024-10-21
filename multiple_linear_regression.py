import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("FuelConsumptionCo2.csv")

# Display the first 10 rows of the dataset
print(df.head(10))

# Show statistical details of the dataset
print(df.describe())

# Select relevant features for analysis
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
          'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head(9))

# Split the dataset into training (80%) and testing (20%) sets
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Create a linear regression model
reg = linear_model.LinearRegression()

# Prepare training data
train_x = np.asanyarray(
    train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# Fit the model to the training data
reg.fit(train_x, train_y)

# Print model intercept and coefficients
print(f"Intercept = {reg.intercept_}")
print(f"Coefficient = {reg.coef_}")

# Prepare testing data and make predictions
y_hat = reg.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_x = np.asanyarray(
    test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Calculate and print the R-squared score
print(f"R^2 Score = {r2_score(test_y, y_hat)}")
