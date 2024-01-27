from Layer import Layer

def test(x,y_true,loss_func,accuracy_func,network:list[Layer],is_classification=False):

    output = x
    for layer in network:
      output = layer.forward(output)
    loss = loss_func(output,y_true)
    if is_classification:
        accuracy = accuracy_func(output,y_true)
        print(f"Loss: {loss} Accuracy: {accuracy*100} %")
    else:
        print(f"Loss: {loss}")