import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()
# x_train = train_data['Cylinders'].to_numpy()
# x_train = train_data['Displacement'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()
# x_test = test_data['Cylinders'].to_numpy()
# x_test = test_data['Displacement'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]

# np.c[...]: concatenation along the second axis (columns)
# np.ones((len(x_train), 1)): column vector (2D array) of ones with the same number of rows as x_train.
X_b_train = np.c_[np.ones((len(x_train), 1)), x_train]

# theta - using formula from lab manual
# np.linalg.inv(): inverse of a matrix
# X_b_train.T: transposed matrix
# .dot: dot product
theta_best = np.linalg.inv(X_b_train.T.dot(X_b_train)).dot(X_b_train.T).dot(y_train)
print("theta_best", theta_best)
# TODO: END

# TODO: calculate error
X_b_test = np.c_[np.ones((len(x_test), 1)), x_test]  # augment feature matrix for test set
# result is a vector of predicted values (y_pred_train) for the training dataset
y_pred_train = X_b_train.dot(theta_best)
y_pred_test = X_b_test.dot(theta_best)

# Mean Squared Error (MSE)
# average of the squares of the differences between the actual (y_train) and predicted (y_pred_train) values
mse_train = np.mean((y_train - y_pred_train) ** 2)
mse_test = np.mean((y_test - y_pred_test) ** 2)

print("mse_train", mse_train)
print("mse_test", mse_test)
# TODO: END

# 1. chart
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# standaryzacja Z - (Z-score normalization)
# µ to średnia z populacji (mean)
mu_train = np.mean(x_train)
# σ to odchylenie standardowe populacji (standard deviation)
# ensuring that the scaled data will have a standard deviation of 1
sigma_train = np.std(x_train)

# the result is a new array, x_train_std, where the data was scaled to have a mean of 0 and a standard deviation of 1
x_train_std = (x_train - mu_train) / sigma_train
x_test_std = (x_test - mu_train) / sigma_train

# augmenting the standardized features with a column of ones
X_b_train_std = np.c_[np.ones((len(x_train_std), 1)), x_train_std]
# TODO: END

# TODO: calculate theta using Batch Gradient Descent
# learning rate for the gradient descent algorithm
# controls how much we adjust the model parameters with respect to the cost gradient
# smaller learning rate requires more iterations to converge to a minimum
# too-large learning rate might overshoot the minimum or cause divergence
learning_rate = 0.0001
# number of times the algorithm will update the model parameters
n_iterations = 100000
m = len(x_train_std)

# θ consists of two parameters because we are considering a simple linear regression
# with one independent variable and an intercept term
theta = np.random.randn(2, 1)  # random initialization

# The Gradient Descent loop
for iteration in range(n_iterations):
    # gradient of the cost function with respect to the model parameters (theta)
    # X_b_train_std.dot(theta) - y_train.reshape(-1, 1): computes the prediction errors by subtracting the actual values from the predictions.
    # X_b_train_std.T.dot(...): Multiplies the transposed augmented feature matrix by the prediction errors, which gives us the sum of the gradient contributions from all training examples.
    # 2/m * ...: Computes the average gradient by dividing by the number of training examples (m) and multiplying by 2, as derived from the cost function's gradient formula.
    gradients = 2/m * X_b_train_std.T.dot(X_b_train_std.dot(theta) - y_train.reshape(-1, 1))
    # update the model parameters by subtracting the product of the learning rate and the gradient from the current values of theta. This step moves the parameters in the direction that minimally decreases the cost function.
    theta -= learning_rate * gradients

print("theta", theta)
# TODO: END

# TODO: calculate error
# Predictions with standardized data
X_b_test_std = np.c_[np.ones((len(x_test_std), 1)), x_test_std]
y_pred_train_std = X_b_train_std.dot(theta)
y_pred_test_std = X_b_test_std.dot(theta)

mse_train_std = np.mean((y_train - y_pred_train_std.flatten()) ** 2)
mse_test_std = np.mean((y_test - y_pred_test_std.flatten()) ** 2)

print("mse_train_std", mse_train_std)
print("mse_test_std", mse_test_std)
# TODO: END

# plot the regression line
x_std = np.linspace(min(x_test_std), max(x_test_std), 100)
x_original = x_std * sigma_train + mu_train  # Convert standardized x values back to original scale

# calculate y values using the theta obtained from Batch Gradient Descent
y = theta[0] + theta[1] * x_std

# 2. chart
# plot the regression line along with the original test data
plt.figure(figsize=(10, 6))
plt.plot(x_original, y, label="Regression Line (Batch Gradient Descent)")
plt.scatter(x_test, y_test, color='red', label="Test Data")
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.title('Linear Regression Model: MPG vs. Weight')
plt.legend()
plt.show()

# x = np.linspace(min(x_test), max(x_test), 100)
# y = float(theta_best[0]) + float(theta_best[1]) * x
# plt.plot(x, y)
# plt.scatter(x_test, y_test)
# plt.xlabel('Weight')
# plt.ylabel('MPG')
# plt.show()
