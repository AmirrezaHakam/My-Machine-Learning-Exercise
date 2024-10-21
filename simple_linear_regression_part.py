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
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print(cdf.head())

# Uncomment to visualize the distribution of features
# plt.figure(1)
# viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# viz.hist()
# plt.show()

# Scatter plot for engine size vs. CO2 emissions
# plt.figure(2)
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("Engine Size (L)")
# plt.ylabel("CO2 Emissions (g/km)")
# plt.title("Engine Size vs CO2 Emissions")
# plt.show()

# Split the dataset into training (80%) and testing (20%)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Uncomment to visualize training and testing data
# plt.figure(3)
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue', label='Training Data')
# plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='red', label='Testing Data')
# plt.xlabel("Engine Size (L)")
# plt.ylabel("CO2 Emissions (g/km)")
# plt.legend()
# plt.title("Training and Testing Data")
# plt.show()

# Create a linear regression model
regr = linear_model.LinearRegression()

# Prepare training data
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# Fit the model to the training data
regr.fit(train_x, train_y)

# Print model coefficients and intercept
print("Coefficient:", regr.coef_)
print("Intercept:", regr.intercept_)

# Uncomment to visualize the regression line
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r', label='Regression Line')
# plt.xlabel("Engine Size (L)")
# plt.ylabel("CO2 Emissions (g/km)")
# plt.title("Regression Line Fit")
# plt.legend()
# plt.show()

# Prepare testing data
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Predict CO2 emissions for the testing set
test_y_ = regr.predict(test_x)

# Calculate and print the R-squared score
print("R^2 Score =", r2_score(test_y, test_y_))
