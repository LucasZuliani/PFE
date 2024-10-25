import torch

class SinActivation(torch.nn.Module):
    def __init__(self) -> None:
        super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, domain_bounds, input_dim=1, hidden_size=20):
        super(FullyConnectedNetwork, self).__init__()
        
        self.lb, self.ub = domain_bounds

        self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc_out = torch.nn.Linear(in_features=hidden_size, out_features=1)

        self.activation1 = SinActivation()
        self.activation2 = torch.nn.Tanh()

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                torch.nn.init.constant_(m.bias, 0.1)

        self.nb_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        
        x_normalised = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        x_normalised = self.activation2(self.fc_in(x_normalised))
        x_normalised = self.activation2(self.fc2(x_normalised))
        x_normalised = self.activation2(self.fc3(x_normalised))
        u_theta = self.fc_out(x_normalised)

        return u_theta