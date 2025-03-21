import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
housing = fetch_california_housing()

# Selecting only one feature (3rd column: "MedInc" - Median Income)
Data_X = housing.data[:, np.newaxis, 2]

# Splitting into training and testing datasets
Data_X_train = Data_X[:-20000]
Data_X_test = Data_X[-20000:]

Data_Y_train = housing.target[:-20000]
Data_Y_test = housing.target[-20000:]

# Create and train the model
model = linear_model.LinearRegression()
model.fit(Data_X_train, Data_Y_train)

# Make predictions
Data_Y_predicted = model.predict(Data_X_test)

# Evaluate the model
mse = mean_squared_error(Data_Y_test, Data_Y_predicted)
r2 = r2_score(Data_Y_test, Data_Y_predicted)

print("Mean Squared Error:", mse)
print("R-squared (R²) Score:", r2)
print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

# Plot actual vs predicted values
#plt.scatter(Data_Y_test, Data_Y_predicted, color='blue', alpha=0.5)
plt.plot([min(Data_Y_test), max(Data_Y_test)], [min(Data_Y_test), max(Data_Y_test)], color='red', linestyle='dashed')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()
