import torch
import matplotlib.pyplot as plt
from NeuralNetworks import Fully_connected_network

if torch.cuda.is_available():
   device = torch.device("cuda")
   print("Running on the GPU")
else:
   device = torch.device("cpu")
   print("Running on the CPU")

def u_exact(x:torch.Tensor):
    u_g = torch.sin(2*x + 1) + 0.2*torch.exp(1.3*x)
    return u_g

torch.manual_seed(12)

x_train = torch.linspace(-1.05, 1.05, 300)[:, None]
u_train = u_exact(x_train)
x_intp = torch.linspace(-1, 1, 200)[:, None]
u_intp = u_exact(x_intp)

lb, ub = x_train.min(), x_train.max()
u_scale = u_train.abs().max()/2

model = Fully_connected_network.FullyConnectedNetwork(domain_bounds=[lb, ub], input_dim=1, hidden_size=20)
criterion = torch.nn.MSELoss()
model_optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)
model_optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), tolerance_change=1e-11, line_search_fn='strong_wolfe')
n_iter_adam = 3000
n_iter_lbfgs = 101

for iter_i in range(n_iter_adam):
    model_optimizer_adam.zero_grad()

    u_pred = model(x_train)
    loss = criterion(u_pred, u_train)*(u_scale**2)
    loss.backward()
    model_optimizer_adam.step()

    if iter_i % 100 == 0:
        print(f"Iteration {iter_i}, loss: {loss.item()}")

def closure_lbfgs():
    model_optimizer_lbfgs.zero_grad()
    u_pred = model(x_train)
    loss = criterion(u_pred, u_train)*(u_scale**2)
    loss.backward()
    return loss

for iter_i in range(n_iter_lbfgs):
    model_optimizer_lbfgs.step(closure_lbfgs)
    if iter_i % 100 == 0:
        print(f"Iteration {iter_i}, loss: {loss.item()}")

u_pred = model(x_intp)
residue_order1 = u_intp - u_pred

plot = True
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(x_intp, u_intp, 'b-', label='Exact')
    axes[0].plot(x_intp, u_pred.detach(), 'r--', label='Predicted')
    axes[0].legend()
    axes[0].set_xlim(lb, ub)
    axes[1].plot(x_intp, residue_order1.detach())
    axes[1].set_xlim(lb, ub)
    plt.show()







