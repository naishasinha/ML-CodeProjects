# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
import numpy as np
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1 / np.sqrt(d)
        self.alpha1 = 1 / np.sqrt(h)
        self.w0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape = torch.Size([h, d])))
        self.b0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape = torch.Size([1, h])))
        self.w1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape = torch.Size([k, h])))
        self.b1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape = torch.Size([1, k])))
        self.params = [self.w0, self.w1, self.b0, self.b1] 
        for param in self.params:
          param.requires_grad = True

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        n,_ = x.shape
        b_0 = self.b0.repeat(n, 1)
        b_1 = self.b1.repeat(n, 1)
        x = torch.matmul(x, self.w0.T) + b_0
        x = torch.nn.functional.relu(x)
        x = torch.matmul(x, self.w1.T) + b_1
        return x


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.alpha0 = 1 / np.sqrt(d)
        self.alpha1 = 1 / np.sqrt(h0)
        self.alpha2 = 1 / np.sqrt(h1)
        self.w0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape = torch.Size([h0, d])))
        self.b0 = torch.nn.Parameter(Uniform(-self.alpha0, self.alpha0).sample(sample_shape = torch.Size([1, h0])))
        self.w1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape = torch.Size([h1, h0])))
        self.b1 = torch.nn.Parameter(Uniform(-self.alpha1, self.alpha1).sample(sample_shape = torch.Size([1, h1])))
        self.w2 = torch.nn.Parameter(Uniform(-self.alpha2, self.alpha2).sample(sample_shape = torch.Size([k, h1])))
        self.b2 = torch.nn.Parameter(Uniform(-self.alpha2, self.alpha2).sample(sample_shape = torch.Size([1, k])))
        self.params = [self.w0, self.w1, self.w2, self.b0, self.b1, self.b2] 
        for param in self.params:
          param.requires_grad = True

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        b_0 = self.b0.repeat(x.shape[0],1)
        x = torch.matmul(x, self.w0.T) + b_0
        x = torch.nn.functional.relu(x)
        b_1 = self.b1.repeat(x.shape[0],1)
        x = torch.matmul(x, self.w1.T) + b_1
        x = torch.nn.functional.relu(x)
        b_2 = self.b2.repeat(x.shape[0],1)
        x = torch.matmul(x, self.w2.T) + b_2
        return x


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    epochs = 32
    losses = []
    accuracies = []
    for i in range(epochs):
        loss_epoch = 0
        accuracy = 0
        for images, labels in train_loader:
            x, y = images, labels
            y_predic = model.forward(x) 
            loss_rand = cross_entropy(y_predic, y)
            optimizer.zero_grad()
            loss_rand.backward()
            optimizer.step()
            predic = torch.argmax(y_predic, 1)
            accuracy += torch.sum(predic == y) / len(predic)
            loss_epoch += loss_rand.item()
        accuracy = accuracy / len(train_loader)
        print(accuracy)
        if accuracy > 0.99:
            break
        losses.append(loss_epoch / len(train_loader))
        accuracies.append(accuracy)
    return losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    model = F1(h = 64, d = 784, k = 10)
    optimizer = Adam(model.params, lr = 5e-3)
    train_loader = DataLoader(TensorDataset(x,y), batch_size = 64, shuffle = True)
    losses = train(model, optimizer, train_loader)
    y_hat = model(x_test)
    test_preds = torch.argmax(y_hat, 1)
    accuracy_value = torch.sum(test_preds == y_test) / len(test_preds)
    print(accuracy_value,'******')
    test_loss_val = cross_entropy(y_hat, y_test).item()
    print(test_loss_val,'******')
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params,'****')

    model = F2(h0 = 32, h1 = 32, d = 784, k = 10)
    optimizer = Adam(model.params, lr = 5e-3)
    train_loader = DataLoader(TensorDataset(x,y), batch_size = 64, shuffle = True)
    losses = train(model, optimizer, train_loader)
    y_hat = model(x_test)
    test_preds = torch.argmax(y_hat,1)
    accuracy_value = torch.sum(test_preds == y_test) / len(test_preds)
    print(accuracy_value,'******')
    test_loss_val = cross_entropy(y_hat, y_test).item()
    print(test_loss_val,'******')
    plt.plot(np.arange(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params,'****')


if __name__ == "__main__":
    main()
