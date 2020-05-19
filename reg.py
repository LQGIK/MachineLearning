import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Select model
model = LinearRegression()

# Read data in from file
with open("data.csv") as f:
    reader = csv.reader(f)
    next(reader)

    claims = []
    payments = []
    for row in reader:
        claims.append(int(row[0]))
        payments.append(float(row[1]))


# Split data
x_testing, x_training,y_testing,y_training = train_test_split(
    claims,payments, test_size = 0.8
)

# Reshape data for 2D array Input
x_training = np.array(x_training).reshape(-1, 1)
x_testing = np.array(x_testing).reshape(-1, 1)
y_testing = np.array(y_testing).reshape(-1, 1)
y_training = np.array(y_training).reshape(-1, 1)


# Fit model to data
model.fit(x_training, y_training)

# Make prediction
predictions = model.predict(x_testing)



# Compute how well we performed
correct = (y_testing == predictions).sum()
incorrect = (y_testing != predictions).sum()
total = len(predictions)



# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_testing, predictions))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_testing, predictions))



# Plot outputs
plt.scatter(x_testing, y_testing,  color='black')
plt.plot(x_testing, predictions, color='blue', linewidth=3)
plt.show()