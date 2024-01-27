
import numpy as np
from Utils import convert_to_one_hot

def mse(y,y_true):
  return np.sum(np.sum((y-y_true)**2,axis=1)) / y.shape[0]

def mse_prime(y,y_true):
  return 2 * (y-y_true)


def softmax(x):
  # Subtract the maximum value for numerical stability
  max_val = np.max(x, axis=1, keepdims=True)
  exp_x = np.exp(x - max_val)

  # Calculate softmax probabilities
  softmax_probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

  return softmax_probs

def cross_entropy_loss(y,y_true):
  epsilon = 1e-15
  y=softmax(y)
  num_classes = y.shape[1]
  y_true = convert_to_one_hot(y_true,num_classes)
  loss = -1 * np.log(y+epsilon) * y_true
  return np.sum(np.sum(loss,axis=1),axis=0) / y.shape[0]

def cross_entropy_loss_prime(y,y_true):
  s = softmax(y)
  num_classes = y.shape[1]
  y_true = convert_to_one_hot(y_true,num_classes)
  return s - y_true

def get_classification_accuracy(output,y_true):
  output = softmax(output)
  correct = 0
  for i in range(output.shape[0]):
    k = np.argmax(output[i])
    if k == y_true[i]:
      correct+=1
  return correct / output.shape[0]