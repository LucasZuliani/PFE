## Libraries ##
import sys, os
import torch
from torch.autograd import grad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from NeuralNetworks import fwdNN as fwd
from NeuralNetworks import residualNN as rnn
from ex2_utils import u_true, du_x1_true, du_x2_true, Corner_Singularity_2D

## Functions ##

# def H1_norm(u: np.ndarray) -> float:
#     tensor_u = torch.tensor(u, dtype=torch.float32, requires_grad=True)

#     l2_norm_u = torch.linalg.norm(tensor_u)
#     grad_u = grad(tensor_u.sum(), tensor_u, create_graph=True)[0]
#     l2_norm_gradu = torch.linalg.norm(grad_u)
#     return torch.sqrt(l2_norm_u**2 + l2_norm_gradu**2).item()

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

def norm_h1(x, u):
    grad_u = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), allow_unused=True, retain_graph=True)[0]
    l2_norm_u = torch.linalg.norm(u)
    l2_norm_gradu = torch.linalg.norm(grad_u)
    return torch.sqrt(l2_norm_u**2 + l2_norm_gradu**2).item()

def assess_solution(model, model_name, grid_size, save=False, df=None):
    evaluation_domain = Corner_Singularity_2D(grid_size=grid_size)
    squared_grid_size = evaluation_domain.squared_grid_size
    evaluation_domain_points = torch.FloatTensor(evaluation_domain.uniform_all_points)
    evaluation_domain_points.requires_grad = True

    model.eval()

    tensor_u_pred = model(evaluation_domain_points)
    tensor_u_exact = u_true(evaluation_domain_points.cpu()).unsqueeze(1)

    norm_l2_u_exact = torch.linalg.norm(tensor_u_exact)
    norm_l2_u_pred = torch.linalg.norm(tensor_u_pred)
    norm_h1_u_exact = norm_h1(evaluation_domain_points, tensor_u_exact)
    norm_h1_u_pred = norm_h1(evaluation_domain_points, tensor_u_pred)
    diff_l2 = torch.linalg.norm(tensor_u_exact - tensor_u_pred)
    diff_h1 = norm_h1(evaluation_domain_points, tensor_u_exact - tensor_u_pred)

    u_pred = (tensor_u_pred.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)
    u_exact = (tensor_u_exact.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)

    tensor_du_pred_x1, tensor_du_pred_x2 = partial_derivative(u=tensor_u_pred,x=evaluation_domain_points)
    tensor_du_exact_x1, tensor_du_exact_x2 = du_x1_true(evaluation_domain_points), du_x2_true(evaluation_domain_points) 
    du_pred_x1, du_pred_x2 = (tensor_du_pred_x1.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1), (tensor_du_pred_x2.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)
    du_exact_x1, du_exact_x2 = (tensor_du_exact_x1.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1), (tensor_du_exact_x2.cpu().detach().numpy()).reshape(squared_grid_size+1, squared_grid_size+1)

    diff_abs = np.abs(u_exact - u_pred)
    diff_abs_dx1 = np.abs(du_exact_x1 - du_pred_x1)
    diff_abs_dx2 = np.abs(du_exact_x2 - du_pred_x2)

    err_relative_l2 = diff_l2 / norm_l2_u_exact
    err_relative_h1 = diff_h1 / norm_h1_u_exact
    

    print(f"Mean absolute difference: {np.mean(diff_abs)}")
    print(f"Norm L2 of the true solution: {norm_l2_u_exact} and of the predicted solution: {norm_l2_u_pred}")
    print(f"Norm H1 of the true solution: {norm_h1_u_exact} and of the predicted solution: {norm_h1_u_pred}")
    print(f"Norm L2 of the relative error: {err_relative_l2}")
    print(f"Norm H1 of the relative error: {err_relative_h1}")

    fig0, axes0 = plt.subplots(1, 3, figsize=(12, 5))
    config_one_plot_im(axes0, fig0, 0, u_exact, 'Exact solution')
    config_one_plot_im(axes0, fig0, 1, u_pred, 'Predicted solution')
    config_one_plot_im(axes0, fig0, 2, diff_abs, 'Absolute difference')
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

    if save:
        # Figures
        if not os.path.exists((f'./Ex2_singularity2D/Figures/{model_name}')):
            os.makedirs(f'./Ex2_singularity2D/Figures/{model_name}')
        fig0.savefig(f'./Ex2_singularity2D/Figures/{model_name}/{model_name}_U_gr{grid_size+1}.png')
        fig1.savefig(f'./Ex2_singularity2D/Figures/{model_name}/{model_name}_dU_dx1_gr{grid_size+1}.png')
        fig2.savefig(f'./Ex2_singularity2D/Figures/{model_name}/{model_name}_dU_dx2_gr{grid_size+1}.png')

        # DataFrame
        key_words = model_name.split('_')
        model_type, model_iter, model_ip_omega, model_ip_boundary, model_beta = key_words[1:6]
        row_to_add = {'Model': model_type, 'Iterations': model_iter, 'IP in Omega': model_ip_omega,
                     'IP on boundary': model_ip_boundary, 'Beta': model_beta[4:], 'Grid size': grid_size+1,
                     'MAE': np.mean(diff_abs), 'L2 relative error': err_relative_l2, 'H1 relative error': err_relative_h1.item()}
        
        df_to_add = pd.DataFrame([row_to_add])
        df = pd.concat([df, df_to_add], ignore_index=True)
        df.drop_duplicates(subset=df.columns[:6].tolist(), inplace=True)
        df.sort_values(by="L2 relative error", ascending=True, inplace=True)
        print(df)
        df.to_csv('./Ex2_singularity2D/ex2_summary.csv', index=False, float_format="%.5f")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=19)
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()

    # model_name = 'ex2_res_iter20k_10000_2500_beta500'
    model_name = 'ex2_res_iter20k_100_75_beta500'
    # model_name = 'ex2_res_iter20k_100_75_beta10000'
    # model_name = 'ex2_res_iter20k_10000_2500_beta10000'
    # model_name = 'ex2_fwd_iter20k_10000_2500_beta500'

    if model_name.split('_')[1] == 'res':
        model = rnn.RitzModel(2)
    else:
        model = fwd.FullyConnectedNetwork(hidden_size=20, input_dim=2)

    model.load_state_dict(torch.load('./Ex2_singularity2D/Models/'+model_name+'.pth', weights_only=True))
    print('\n')
    print(f'Number of parameters: {model.nb_params}')
    df = pd.read_csv('./Ex2_singularity2D/ex2_summary.csv', sep=',')
    assess_solution(model, model_name, args.grid_size, args.save, df)