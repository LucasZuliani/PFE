import torch

class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_size=20):
        super(FullyConnectedNetwork, self).__init__()
        
        self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc_out = torch.nn.Linear(in_features=hidden_size, out_features=1)

        self.activation = torch.nn.Tanh()

    def forward(self, x):

        x = self.activation(self.fc_in(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        u_theta = self.fc_out(x)

        return u_theta