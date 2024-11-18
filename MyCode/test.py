import torch

class NN(torch.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = torch.nn.Linear(2, 4)
        self.fc2 = torch.nn.Linear(4, 4)
        self.fc3 = torch.nn.Linear(4, 1)

        self.conv1d = torch.nn.Conv1d(1, 1, 1)

        self.activation = torch.nn.Tanh()
    def forward(self, x):
        # print(f'Input size: {x.shape}')
        x = self.activation(self.fc1(x))
        # print(f'Size after fc1: {x.shape}')
        x = self.fc2(x)
        # print(f'Size after fc2: {x.shape}')
        x = self.fc3(x).unsqueeze(1)
        # print(f'Ouput size: {x.shape}')
        return x
    
if __name__ == '__main__':
    x = torch.randn(1, 10, 2).permute(0, 2, 1)
    y1 = torch.nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)(x)
    y2 = torch.nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, padding=1)(y1)
    y3 = torch.nn.Conv1d(in_channels=8, out_channels=2, kernel_size=1)(y2)
    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    # model = NN()
    # x = torch.rand(10, 2)
    # y = model(x)
