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

model1 = Fully_connected_network.FullyConnectedNetwork(domain_bounds=[lb, ub], kappa=1, input_dim=1, hidden_size=20, actv=0)
criterion = torch.nn.MSELoss()
model1_optimizer_adam = torch.optim.Adam(model1.parameters(), lr=0.001)
model1_optimizer_lbfgs = torch.optim.LBFGS(model1.parameters(), tolerance_change=1e-11, line_search_fn='strong_wolfe')
n_iter_adam = 3000
n_iter_lbfgs = 101

for iter_i in range(n_iter_adam):
    model1_optimizer_adam.zero_grad()

    u_pred = model1(x_train)
    loss = criterion(u_pred, u_train)*(u_scale**2)
    loss.backward()
    model1_optimizer_adam.step()

    if iter_i % 100 == 0:
        print(f"Iteration {iter_i}, loss: {loss.item()}")

def closure_lbfgs():
    model1_optimizer_lbfgs.zero_grad()
    u_pred = model1(x_train)
    loss = criterion(u_pred, u_train)*(u_scale**2)
    loss.backward()
    return loss

for iter_i in range(n_iter_lbfgs):
    model1_optimizer_lbfgs.step(closure_lbfgs)
    if iter_i % 100 == 0:
        print(f"Iteration {iter_i}, loss: {loss.item()}")

u_pred = model1(x_intp)
residue_order1 = u_intp - u_pred

# Second stage of training
u_train2 = residue_order1.detach()
nb_zeros = torch.where(u_train2[:-1, 0] * u_train2[1:, 0] < 0)[0]
kappa2 = 3*(nb_zeros.shape[0])
u_scale = u_train2.abs().max()/2

model2 = Fully_connected_network.FullyConnectedNetwork(domain_bounds=[lb, ub], kappa=kappa2, input_dim=1, hidden_size=20, actv=1)
model2_optimizer_adam = torch.optim.Adam(model2.parameters(), lr=0.001)
model2_optimizer_lbfgs = torch.optim.LBFGS(model2.parameters(), tolerance_change=1e-11, line_search_fn='strong_wolfe')

n_iter_adam = 5000

for iter_i in range(n_iter_adam):
    model2_optimizer_adam.zero_grad()

    u_pred2 = model2(u_train2)
    loss = criterion(u_pred2, u_train2)*(u_scale**2)
    loss.backward()
    model2_optimizer_adam.step()

    if iter_i % 100 == 0:
        print(f"Iteration {iter_i}, loss: {loss.item()}")

redisue_order1_pred = model2(residue_order1)
plot = True
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(x_intp, u_intp, 'b-', label='Exact')
    axes[0].plot(x_intp, u_pred.detach(), 'r--', label='Predicted')
    axes[0].legend()
    axes[0].set_xlim(lb, ub)

    axes[1].plot(x_intp, residue_order1.detach(), 'b-', label=r'$e_1(x) = u_g(x) - u_0(x)$')
    axes[1].plot(x_intp, redisue_order1_pred.detach(), 'r--', label=r'2nd-stage NN: $u_1(x)$')
    axes[1].set_xlim(lb, ub)

    plt.show()







