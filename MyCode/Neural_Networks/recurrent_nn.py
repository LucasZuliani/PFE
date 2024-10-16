import torch 

class ActivationCube(torch.nn.Module):
    def __init__(self) -> None:
        super(ActivationCube, self).__init__()

    def forward(self, x):
        return torch.maximum(x**3, torch.tensor(0.))

class RecurentBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super(RecurentBlock, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.activation1 = ActivationCube()
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.activation2 = ActivationCube()
    
    def forward(self, x):
        retain_x = x
        x = self.fc1(x)
        x = self.activation1(x)
        x = self.fc2(x)
        x = self.activation2(x)
        x = x + retain_x
        return x

class RitzModel(torch.nn.Module):

    def __init__(self, num_blocks=4, hidden_size=10):
        super(RitzModel, self).__init__()

        self.fc_in = torch.nn.Linear(1, hidden_size)
        self.blocks = torch.nn.ModuleList([RecurentBlock(hidden_size=hidden_size) for _ in range(num_blocks)])
        self.fc_out = torch.nn.Linear(hidden_size, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)

    def forward(self, x):
        
        x = self.fc_in(x)
        for block in self.blocks:
            x = block(x)
        u_theta = self.fc_out(x)

        return u_theta