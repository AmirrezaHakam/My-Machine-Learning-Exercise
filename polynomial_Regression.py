from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

df = pd.read_csv("FuelConsumptionCo2.csv")
df.head()
df.describe()

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emission")
# plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=3)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
clf.fit(train_x_poly, train_y)
print(f"InterCept = {clf.intercept_}")
print(f"Coefficient = {clf.coef_}")

plt.figure(2)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
xx = np.arange(0, 10, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1]*xx + clf.coef_[0][2] * \
    np.power(xx, 2)+clf.coef_[0][3]*np.power(xx, 3)
plt.plot(xx, yy)
plt.show()

test_x_poly = poly.fit_transform(test_x)
test_y_y = clf.predict(test_x_poly)
print(f"R2 Score = {r2_score(test_y,test_y_y)}")
