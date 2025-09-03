# Import required libraries
import urllib2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import certifi
import ssl

# URL of the dataset (Boston Housing dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# Create an unverified context
context = ssl._create_unverified_context()

# Download the data
response = urllib2.urlopen(url, context=context)
data = response.read()

# Save the data to a local file
with open("housing.data", "w") as f:
    f.write(data)

# Load the data into a pandas DataFrame
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
df = pd.read_csv("housing.data", delim_whitespace=True, names=column_names)

# Display the first few rows of the data
print df.head()


# Select features and target variable
X = df[['RM', 'LSTAT', 'PTRATIO']]  # Features: average number of rooms, % lower status of the population, pupil-teacher ratio
y = df['MEDV']  # Target: Median value of owner-occupied homes in $1000s

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print "Training set shape:", X_train.shape
print "Testing set shape:", X_test.shape



# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print "Mean squared error: ", mse
print "Mean absolute error: ", mae
print "R-squared score: ", r2

# Print model coefficients
for feature, coef in zip(X.columns, model.coef_):
    print("{}: {}".format(feature, coef))


# Scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Housing Prices')
plt.show()