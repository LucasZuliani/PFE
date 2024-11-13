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
import plots
from ex2_utils import u_true, du_x1_true, du_x2_true, Corner_Singularity_2D

## Functions ##

def assess_solution(model, model_name, grid_size, save=False, df=None):
    evaluation_domain_points = plots.create_uniform_grid((grid_size, grid_size), (-1, -1), (1, 1))
    evaluation_domain_points.requires_grad = True

    model.eval()

    u_pred = model(evaluation_domain_points)
    u_exact = u_true(evaluation_domain_points.cpu()).unsqueeze(1)
    u_pred_reshaped = plots.reshape_tensor_to_cpu(u_pred, grid_size)
    u_exact_reshaped = plots.reshape_tensor_to_cpu(u_exact, grid_size)

    du_pred_x1, du_pred_x2 = plots.partial_derivative_2D(u=u_pred,x=evaluation_domain_points)
    du_exact_x1, du_exact_x2 = du_x1_true(evaluation_domain_points), du_x2_true(evaluation_domain_points) 
    du_pred_x1_reshaped, du_pred_x2_reshaped = plots.reshape_tensor_to_cpu(du_pred_x1, grid_size), plots.reshape_tensor_to_cpu(du_pred_x2, grid_size)
    du_exact_x1_reshaped, du_exact_x2_reshaped = plots.reshape_tensor_to_cpu(du_exact_x1, grid_size), plots.reshape_tensor_to_cpu(du_exact_x2, grid_size)

    mae = plots.compute_mae(u_pred, u_exact)
    err_relative_l2 = plots.compute_l2_relative_error(u_pred, u_exact)
    err_relative_h1 = plots.compute_h1_relative_error(u_pred, u_exact, evaluation_domain_points)

    fig0 = plots.plot_config1(u_pred_reshaped, u_exact_reshaped)
    fig1 = plots.plot_config1(du_pred_x1_reshaped, du_exact_x1_reshaped)
    fig2 = plots.plot_config1(du_pred_x2_reshaped, du_exact_x2_reshaped)

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
                     'MAE': mae, 'L2 relative error': err_relative_l2, 'H1 relative error': err_relative_h1}
        
        df_to_add = pd.DataFrame([row_to_add])
        df = pd.concat([df, df_to_add], ignore_index=True)
        df.drop_duplicates(subset=df.columns[:6].tolist(), inplace=True)
        df.sort_values(by="L2 relative error", ascending=True, inplace=True)
        df.to_csv('./Ex2_singularity2D/ex2_summary.csv', index=False, float_format="%.5f")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=19)
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()

    # model_name = 'ex2_res_iter20k_10000_2500_beta500'
    # model_name = 'ex2_res_iter20k_100_75_beta500'
    # model_name = 'ex2_res_iter20k_100_75_beta10000'
    # model_name = 'ex2_res_iter20k_10000_2500_beta10000'
    model_name = 'ex2_fwd_iter20k_10000_2500_beta500'

    if model_name.split('_')[1] == 'res':
        model = rnn.RitzModel(2)
    else:
        model = fwd.FullyConnectedNetwork(hidden_size=20, input_dim=2)

    model.load_state_dict(torch.load('./Ex2_singularity2D/Models/'+model_name+'.pth', weights_only=True))
    print('\n')
    print(f'Number of parameters: {model.nb_params}')
    df = pd.read_csv('./Ex2_singularity2D/ex2_summary.csv', sep=',')
    assess_solution(model, model_name, args.grid_size, args.save, df)