from Layer import Layer

def train(x,y_true,loss_func,loss_prime,accuracy_func,network:list[Layer],is_classification=False,epochs=10000,learning_rate=0.0001):

  for epoch in range(epochs):
    output = x
    for layer in network:
      output = layer.forward(output)
    loss = loss_func(output,y_true)
    if is_classification:
        accuracy = accuracy_func(output,y_true)
        print(f"Epoch {epoch} Loss: {loss} Accuracy: {accuracy*100} %")
    else:
        print(f"Epoch {epoch} Loss: {loss}")

    out_grad = loss_prime(output,y_true)
    l = len(network)
    for i in range(l-1,-1,-1):
      out_grad = network[i].back_propogate(out_grad,learning_rate)