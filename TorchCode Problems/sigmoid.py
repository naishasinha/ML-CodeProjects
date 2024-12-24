class Sigmoid(torch.nn.Module):
    '''
    Input: A PyTorch tensor x with arbritrary shape
    Output: A PyTorch tensor of the same shape, containing the sigmoid transformation of each element in x.
    
    CONSTRAINTS
    - Cannot use built-in sigmoid functions from PyTorch
    - Should handle tensors with high dimensions (e.g. 3D or more)
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the sigmoid function manually
        return 1 / (1 + torch.exp(-x))
