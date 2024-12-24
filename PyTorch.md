# PyTorch
**Pytorch Documentation:** https://pytorch.org/docs/stable/index.html 
***
* Library for programming with tensors - multidimensional arrays that represent data and parameters in deep neural networks
* Help train ML models using Python
* Facilitates high performance computing on a GPU
* Supports a dynamic computation graph (allows models to be optimized at runtime(. Does this by constructing a DAG consisting of functions that keeps track of all the executed operations on the tensors, allowing you to change the shape, size, and operations after every iteration if needed.

Tensor is similar to a multi-dimensional array, creates a 2D array or Matrix with Python, then use torch to convert it into a tensor:
```
data = [[1, 2, 3, 4,], [3, 4, 5, 6]]
x_data = torch.tensor(data)
```

![image](https://github.com/user-attachments/assets/462eb2d5-a58b-45ff-94dc-209e76672b69)

![image](https://github.com/user-attachments/assets/53a53294-b381-4c04-aa72-4bd685a78890)

## Other Computations/Uses
```
# Convert all integers into random floating points
x_rand = torch.rand_like(x_data, dtype = torch.float)

# Linear Algebra: take multiple tensors and multiply them together
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)

result = torch.matmul(tensor1, tensor2)

# Build a Deep Neural Network

# Image Classifier: create a class that inherits from the neural network module class
class imgClass(nn.Module):
  def __init__(self):
  # Inside constructor, build NN layer by layer
  super().__init__()
  self.flatten = nn.Flatten() # takes multi-dimensional input (img) and convert it into 1D
  self.linear_relu_stack = nn.Sequential(
    nn.Linear(28*28, 512), # Fully-connected Layer
    nn.ReLU(), # Layer
    nn.Linear(512, 512), # Layer: each node is a mini statistical model
    nn.ReLU(),
    nn.Linear(512, 10),
  )

# Forward method describes the flow of data
def forward(self, x):
  x = self.flatten(x)
  logits = self.linear_relu_stack(x)
  return logits

# Instantiate model to GPU
model = HiMom().to("cuda")
X = some_data
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)

print(f"And my prediction is... {y_pred}")

```
