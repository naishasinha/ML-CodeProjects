# Activation Functions (Notes)

## 1. Whats/Wheres/Hows for Activation Functions

### 1.1 Neural Networks
* Family of model architectures designed to find nonlinear patterns in data. Non-linear: you can't use a linear function to cleanly separate different features
* Weights (learnable parameter): determine the strength of connections between neurons, and the importance of input features on the output. Weights are learned during the training process and applied to input features to determine their impact on the output. (In _y = mx + b_, _m_ is the weight, as it is applied to the input.
* Biases (learnable parameter): allows the model to adjust the output independently of the input features, providing a baseline output. They are an additional input into the next layer that guarantees that there will still be an activation in the neuron, even when all the inputs are zeros.

![image](https://github.com/user-attachments/assets/4e62bcc9-60fb-45a7-bded-b29c76edd19b)


![image](https://github.com/user-attachments/assets/a4d8dd48-a927-44d5-97aa-5020ffc59908)


![image](https://github.com/user-attachments/assets/6fe62659-af2e-41d8-a2a6-3119e565b26d)

### 1.2 Activation Function
Enables neural networks to learn nonlinear (complex, quadratic) relationships between features and the label. **Think**: straight line vs. curve.
* Sigmoid
* tanh
* ReLU (helps with vanishing gradients)
* Leaky ReLU (helps with blocked info < 0)
* Maxout
* ELU (helps with blocked info < 0)

![image](https://github.com/user-attachments/assets/27e8a5d6-2fbd-4f94-836b-cf695604c3b2)


## 2. Details With Examples

Activation of output neurons:
* 2 neurons = 2 classes - classification: softmax
* for regression: don't need activation function

More complicated data = More layers
Vanishing gradients = no learning. (gradient descent minimizes the error between predicted and actual results by iteratively adjusting a model's parameters. The algorithm uses a cost function to measure the model's accuracy with each parameter update)

