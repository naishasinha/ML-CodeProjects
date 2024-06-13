from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    vector = (X @ weight - y + bias)    
    updated_weight = weight - 2 * eta * (X.T @ vector)
    updated_bias = bias - 2 * eta * vector.sum()    

    updated_weight[np.abs(updated_weight) <= 2 * eta * _lambda] = 0
    updated_weight[updated_weight < -2*eta*_lambda] += 2* eta *_lambda
    updated_weight[updated_weight > 2*eta*_lambda] -= 2 * eta * _lambda    
    return updated_weight, updated_bias



@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    # loss function
    n, d = X.shape        

    v = (X @ weight - y + bias)    
    return (v @ v) + _lambda * np.linalg.norm(weight, ord=1)


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: Optional[np.ndarray] = None
    b = start_bias
    w = start_weight
    while old_w is None or not convergence_criterion(w, old_w, b, old_b, convergence_delta):
        old_w = np.copy(w)
        old_b = np.copy(b)
        w, b = step(X, y, w, b, _lambda, eta)
        max_delta = np.abs(np.hstack((w - old_w, b - old_b))).max()
    return w, b



@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    return np.abs(weight - old_w).max() <= convergence_delta


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    n = 500
    d = 1000
    k = 100
    w_true = np.hstack((np.linspace(1/k, 1, num=k), np.zeros(d-k)))    

    random_gaussian_noise = np.random.default_rng()
    X = random_gaussian_noise.normal(size=(n, d))
    e = random_gaussian_noise.normal(size=n)
    y = X @ w_true + e

    _lambda = 2*np.abs(X.T @ (y - y.mean())).max()
    nonZero = 0
    lambdas, num_non_zeros = [], []
    fdr, tpr = [], []
    while nonZero < d:
        w, b = train(X, y, _lambda, eta=2e-5, convergence_delta=1e-4)                
        nonZero = np.count_nonzero(w)
        lambdas.append(_lambda)
        num_non_zeros.append(nonZero)

        np.count_nonzero(w[0:k])
        if nonZero == 0:
            fdr.append(0)
            tpr.append(0)
        else:
            fdr.append(np.count_nonzero(w[k:d]) / nonZero)
            tpr.append(np.count_nonzero(w[0:k]) / k)
        _lambda *= 0.5
    
    plt.figure(figsize = (10,5))
    plt.plot(lambdas, num_non_zeros)
    plt.xscale('log')
    plt.title('Plot 1: Non-zero Count V. Lambda')
    plt.xlabel('lambda')
    plt.ylabel('nonzero weight elements')
    plt.show()

    plt.figure(figsize = (10,5))
    plt.plot(fdr, tpr)
    plt.title('Plot 2: tpr V. fdr')
    plt.xlabel('False Discovery Rate (FDR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.show()

if __name__ == "__main__":
    main()
