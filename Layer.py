import numpy as np

class Layer():
  def forward():
    pass

  def back_propogate():
    pass


class DenseLayer(Layer):
  def __init__(self,input_size,output_size):
    self.weight = np.random.rand(input_size,output_size)
    self.bias = np.random.rand(1,output_size)
    # print(self.weight)
    # print(self.bias)
  def forward(self,input):
    self.input = input
    return input @ self.weight + self.bias

  def back_propogate(self,out_grad,learning_rate=0.1):
    bias_grad = np.sum(out_grad,axis=0) / out_grad.shape[0]
    weight_grad = self.input.T @ out_grad / out_grad.shape[0]
    input_grad = out_grad @ self.weight.T
    # print(f"out_grad shape: {out_grad.shape}")
    # print(f"input shape:{self.input.shape}")
    # print(f"weight_grad shape: {weight_grad.shape}")
    self.weight-=learning_rate * weight_grad
    self.bias-=learning_rate * bias_grad
    return input_grad

class ReLU(Layer):
  def forward(self,input):
    self.input = input
    return np.where(input < 0 ,0,input)

  def back_propogate(self,out_grad,learning_rate=0.1):
    o= out_grad * np.where(self.input<=0,0,1)
    # print(f"in grad at relu = {o}")
    return o