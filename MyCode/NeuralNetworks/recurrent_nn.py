import torch

class ActivationCube(torch.nn.Module):
    """
    Activation function that returns the cube of the input if it is positive, and 0 otherwise, used in the paper for Deep Ritz method.

    >>> activation = ActivationCube()
    >>> activation(torch.tensor(2.))
    tensor(8.)
    >>> activation(torch.tensor(-2.))
    tensor(0.)
    """
    def __init__(self) -> None:
        super(ActivationCube, self).__init__()

    def forward(self, x):
        return torch.maximum(x**3, torch.tensor(0.))

class RecurrentBlock(torch.nn.Module):
    def __init__(self, in_size, hidden_size):
        super(RecurrentBlock, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(in_features=in_size, out_features=hidden_size)
        self.activation1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.activation2 = torch.nn.Tanh()
    
    def forward(self, x):
        if self.in_size < self.hidden_size:
            padding = (0, self.hidden_size - self.in_size)
            retain_x = torch.nn.functional.pad(x, padding, mode='constant', value=0)
        else:
            retain_x = x
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = x + retain_x
        return x

class RitzModel(torch.nn.Module):
    def __init__(self, input_dim=2, num_blocks=4, hidden_size=10):
        super(RitzModel, self).__init__()

        self.blocks = torch.nn.ModuleList([
            RecurrentBlock(in_size=input_dim if block_i == 0 else hidden_size, hidden_size=hidden_size)
            for block_i in range(num_blocks)])
        self.fc_out = torch.nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)

        self.nb_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
                
        for block in self.blocks:
            x = block(x)
        u_theta = self.fc_out(x)

        return u_theta