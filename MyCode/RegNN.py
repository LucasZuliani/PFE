import torch
import matplotlib.pyplot as plt
from NeuralNetworks import Fully_connected_network

def u_exact(x:torch.Tensor):
    u_g = torch.sin(2*x + 1) + 0.2*torch.exp(1.3*x)
    return u_g

x_train = torch.linspace(-1.05, 1.05, 300)[:, None]
u_train = u_exact(x_train)

lb, ub = x_train.min(), x_train.max()

model = Fully_connected_network.FullyConnectedNetwork(domain_bounds=[lb, ub], input_dim=1, hidden_size=20)
print(model.nb_params)
# print(lb, ub)
# Plot the exact solution
# plt.figure(figsize=(15, 6))
# plt.plot(x_train, u_train)
# plt.show()






