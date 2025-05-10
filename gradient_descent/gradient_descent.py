import numpy as np
weights = np.array([1, 2])
input_data = np.array([3, 4])
target = 6

# learning_rate using for updating weights by gradient descent
learning_rate = 0.01

# Calculate the prediction
preds = (weights * input_data).sum()

# Calculate the error
error = preds - target

# First error is 5 (We need 0 for perfect prediction)
print("error: " + str(error))


gradient = 2 * input_data * error
print("gradient: " + str(gradient))

# Calculate the updated weights by subtracting the gradient from the weights
weights_updated = weights - learning_rate * gradient

preds_updated = (weights_updated * input_data).sum()
error_updated = preds_updated - target

# Second error is 2.5 after updating the weights
print("error_updated: " + str(error_updated))