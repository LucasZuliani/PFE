## Libraries ##
import sys, os
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
# warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from NeuralNetworks import recurrent_nn as rnn
from ex2_utils import u_true, Corner_Singularity_2D

## Functions ##

def H1_norm(u):
    grad_ux, grad_uy = np.gradient(u)
    grad_norm_squared = grad_ux**2 + grad_uy**2
    return np.sqrt(np.sum(grad_norm_squared) + np.sum(u**2))

def partial_derivative(u, x):
    assert u.shape[0] == x.shape[0]
    assert x.shape[1] == 2
    print("\033[92mGradient computation: dim ok\033[0m")

    grad_u = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), allow_unused=True)[0]
    grad_ux1, grad_ux2 = grad_u[:, 0], grad_u[:, 1]
    return grad_ux1, grad_ux2

def config_one_plot_im(axes, fig, inds, to_plot, title):
    if isinstance(inds, int):
        ind = inds
        im = axes[ind].imshow(to_plot, cmap='jet', extent=[-1, 1, -1, 1], interpolation='bicubic')
        axes[ind].set_title(title)
        fig.colorbar(im, ax=axes[ind], shrink=0.6)
    else:
        ind_i, ind_j = inds
        im = axes[ind_i, ind_j].imshow(to_plot, cmap='jet', extent=[-1, 1, -1, 1], interpolation='bicubic')
        axes[ind_i, ind_j].set_title(title)
        fig.colorbar(im, ax=axes[ind_i, ind_j], shrink=0.6)
    return 

def assess_solution(model, model_name):
    evaluation_domain = Corner_Singularity_2D()
    squared_grid_size = evaluation_domain.squared_grid_size
    evaluation_domain_points = torch.FloatTensor(evaluation_domain.uniform_all_points)
    evaluation_domain_points.requires_grad = True

    model.eval()

    tensor_u_pred = model(evaluation_domain_points)
    tensor_u_exact = u_true(evaluation_domain_points)
    u_pred = (tensor_u_pred.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)
    u_exact = (tensor_u_exact.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)

    tensor_du_pred_x1, tensor_du_pred_x2 = partial_derivative(u=tensor_u_pred,x=evaluation_domain_points)
    tensor_du_exact_x1, tensor_du_exact_x2 = partial_derivative(u=tensor_u_exact,x=evaluation_domain_points)  
    du_pred_x1, du_pred_x2 = (tensor_du_pred_x1.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1), (tensor_du_pred_x2.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)
    du_exact_x1, du_exact_x2 = (tensor_du_exact_x1.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1), (tensor_du_exact_x2.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)

    diff_abs = np.abs(u_exact - u_pred)
    diff_abs_dx1 = np.abs(du_exact_x1 - du_pred_x1)
    diff_abs_dx2 = np.abs(du_exact_x2 - du_pred_x2)

    diff_l2 = np.linalg.norm(diff_abs.flatten())
    diff_h1 = H1_norm(u_exact - u_pred)

    err_l2_pred, err_l2_true_sol = np.linalg.norm(u_pred.flatten()), np.linalg.norm(u_exact.flatten())
    err_h1_pred, err_h1_true_sol = H1_norm(u_pred), H1_norm(u_exact)
    err_relative_l2 = diff_l2 / np.linalg.norm(u_exact.flatten())
    err_relative_h1 = diff_h1 / H1_norm(u_exact)
    

    print(f"Mean absolute difference: {np.mean(diff_abs)}")
    print(f"Norm L2 of the true solution: {err_l2_true_sol} and of the predicted solution: {err_l2_pred}")
    print(f"Norm H1 of the true solution: {err_h1_true_sol} and of the predicted solution: {err_h1_pred}")
    print(f"Norm L2 of the relative error: {err_relative_l2}")
    print(f"Norm H1 of the relative error: {err_relative_h1}")

    fig0, axes0 = plt.subplots(1, 3, figsize=(12, 5))
    config_one_plot_im(axes0, fig0, 0, u_exact, 'Exact solution')
    config_one_plot_im(axes0, fig0, 1, u_pred, 'Predicted solution')
    config_one_plot_im(axes0, fig0, 2, diff_abs, 'Absolute difference')
    fig0.savefig(f'./Ex2_singularity2D/Figures/{model_name}_U.png')
    plt.tight_layout()
    plt.show()

    fig1, axes1 = plt.subplots(1, 3, figsize=(12, 5))
    config_one_plot_im(axes1, fig1, 0, du_exact_x1, 'Exact du/dx1')
    config_one_plot_im(axes1, fig1, 1, du_pred_x1, 'Predicted du/dx1')
    config_one_plot_im(axes1, fig1, 2, diff_abs_dx1, 'Absolute difference du/dx1')
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 5))
    config_one_plot_im(axes2, fig2, 0, du_exact_x2, 'Exact du/dx2')
    config_one_plot_im(axes2, fig2, 1, du_pred_x2, 'Predicted du/dx2')
    config_one_plot_im(axes2, fig2, 2, diff_abs_dx2, 'Absolute difference du/dx2')
    plt.tight_layout()
    plt.show()

    # fig3, axes3 = plt.subplots(2, 2, figsize=(12, 5))
    # axes3[0, 0].plot(evaluation_domain_points[:20, 0].detach().numpy(), tensor_du_exact_x1[:20].detach().numpy(), label='Exact du/dx1')
    # axes3[0, 1].plot(evaluation_domain_points[:20, 0].detach().numpy(), tensor_du_pred_x1[:20].detach().numpy(), label='Predicted du/dx1')
    # plt.tight_layout()
    # plt.show()
    
if __name__=="__main__":
    model_name = 'ex2_res_iter20k_10k_2500_beta500'
    model = rnn.RitzModel(2).to(('cpu'))
    model.load_state_dict(torch.load('./Ex2_singularity2D/Models/'+model_name+'.pth', weights_only=True))
    print('\n')
    assess_solution(model, model_name)