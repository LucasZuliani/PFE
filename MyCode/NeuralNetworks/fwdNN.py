import torch

class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, hidden_size=20, n_hidden_layers=2, input_dim=2):
        super(FullyConnectedNetwork, self).__init__()

        self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=hidden_size, out_features=hidden_size) for _ in range(n_hidden_layers)]
        )
        self.fc_out = torch.nn.Linear(in_features=hidden_size, out_features=1)

        self.activation = torch.nn.Tanh()
        
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)

        self.nb_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.activation(self.fc_in(x))

        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        u_theta = self.fc_out(x)

        return u_theta