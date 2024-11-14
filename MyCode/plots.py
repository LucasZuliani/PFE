import torch
from torch.autograd import grad
import fenics
import matplotlib.pyplot as plt
import numpy as np

def norm_h1(x: torch.Tensor, u: torch.Tensor) -> float:
    grad_u = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), allow_unused=True, retain_graph=True)[0]
    l2_norm_u = torch.linalg.norm(u)
    l2_norm_gradu = torch.linalg.norm(grad_u)
    return torch.sqrt(l2_norm_u**2 + l2_norm_gradu**2).item()

def norm_l2(u: torch.Tensor) -> float:
    return torch.linalg.norm(u).item()

def partial_derivative_2D(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    assert u.shape[0] == x.shape[0]
    assert x.shape[1] == 2
    print("\033[92mGradient computation: dim ok\033[0m")

    grad_u = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), allow_unused=True, retain_graph=True)[0]
    grad_ux1, grad_ux2 = grad_u[:, 0], grad_u[:, 1]
    return grad_ux1, grad_ux2

def create_uniform_grid(grid_size: tuple, lower_bound_xy: tuple, upper_bound_xy: tuple) -> torch.Tensor:
    mesh = fenics.RectangleMesh(fenics.Point(lower_bound_xy[0], lower_bound_xy[1]), fenics.Point(upper_bound_xy[0], upper_bound_xy[1]), grid_size[0], grid_size[1])
    mesh_coordinates = mesh.coordinates()
    return torch.FloatTensor(mesh_coordinates)

def compute_mae(u_pred: torch.Tensor, u_exact: torch.Tensor) -> torch.Tensor:
    mae = torch.mean(torch.abs(u_pred - u_exact))
    print(f'Mean absolute error: {mae}')
    return mae.item()

def compute_l2_relative_error(u_pred: torch.Tensor, u_exact: torch.Tensor) -> torch.Tensor:
    l2_error = norm_l2(u_exact - u_pred)
    l2_u_exact = norm_l2(u_exact)
    l2_u_pred = norm_l2(u_pred)
    l2_relative = l2_error / l2_u_exact
    print(f'l2_relative_error: {l2_relative}, l2_u_true: {l2_u_exact}, l2_u_pred: {l2_u_pred}')
    return l2_relative

def compute_h1_relative_error(u_pred: torch.Tensor, u_exact: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    h1_error = norm_h1(x, u_exact - u_pred)
    h1_u_exact = norm_h1(x, u_exact)
    h1_u_pred = norm_h1(x, u_pred)
    h1_relative = h1_error / h1_u_exact
    print(f'h1_relative_error: {h1_relative}, h1_u_true: {h1_u_exact}, h1_u_pred: {h1_u_pred}')
    return h1_relative

def reshape_tensor_to_cpu(tensor: torch.Tensor, grid_size: int) -> np.ndarray:
    tensor = tensor.cpu().detach().numpy()
    return tensor.reshape(grid_size+1, grid_size+1)

def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.cpu().detach()

def plot_config1(u_pred: torch.Tensor, u_exact: torch.Tensor, title: str) -> plt.Figure:

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    im0 = ax[0].imshow(u_exact, cmap='jet', extent=[-1, 1, -1, 1], interpolation='bicubic')
    ax[0].set_title(title  + ' exact')
    fig.colorbar(im0, ax=ax[0], shrink=0.6)

    im1 = ax[1].imshow(u_pred, cmap='jet', extent=[-1, 1, -1, 1], interpolation='bicubic')
    ax[1].set_title(title + ' predicted ')
    fig.colorbar(im1, ax=ax[1], shrink=0.6)

    im2 = ax[2].imshow(u_exact - u_pred, cmap='jet', extent=[-1, 1, -1, 1], interpolation='bicubic')
    ax[2].set_title('Error')
    fig.colorbar(im2, ax=ax[2], shrink=0.6)
    plt.tight_layout()
    plt.show()

    return fig
