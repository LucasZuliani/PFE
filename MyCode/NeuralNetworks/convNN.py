import torch

class ConvolutionalNetwork(torch.nn.Module):
    def __init__(self, hidden_size=20, input_dim=2):
        super(ConvolutionalNetwork, self).__init__()

        self.fc_in = torch.nn.Linear(in_features=input_dim, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.conv1 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=2*hidden_size, kernel_size=3, padding=1)

        self.conv_out = torch.nn.Conv1d(in_channels=2*hidden_size, out_channels=2, kernel_size=1)
        self.fc_out = torch.nn.Linear(in_features=2, out_features=1)

        self.dropout = torch.nn.Dropout(p=0.2)
    
        self.activation = torch.nn.Tanh()

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, torch.nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = self.activation(self.fc_in(x))
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = (x.unsqueeze(0)).permute(0, 2, 1)

        x = self.conv1(x)
        x = self.conv_out(x)

        u_theta = self.fc_out(x.squeeze(0).permute(1, 0))
        return u_theta
