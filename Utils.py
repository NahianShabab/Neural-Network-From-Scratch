import numpy as np

def convert_to_one_hot(y_true, num_classes):
  # Ensure y_true is a 1D array
  y_true = np.squeeze(y_true)

  # Create an identity matrix of size num_classes
  identity_matrix = np.eye(num_classes)

  # Use y_true as indices to get the one-hot encoded matrix
  y_one_hot = identity_matrix[y_true]

  return y_one_hot