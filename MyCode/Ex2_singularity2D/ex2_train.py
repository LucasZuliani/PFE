import ex2_utils
import torch
from torch.autograd import grad
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from NeuralNetworks import recurrent_nn as rnn

## Loss function ##

class DeepRitzLoss(torch.nn.Module):
    def __init__(self):
        super(DeepRitzLoss, self).__init__()
        self.mse = torch.nn.MSELoss()
    
    def forward(self, x_in_omega, output_ux_in_omega, true_operator_x, output_on_boundary, true_value_on_boundary, boundary_regulizer=500):
        grad_u = grad(inputs=x_in_omega, outputs=output_ux_in_omega, grad_outputs=torch.ones_like(output_ux_in_omega), create_graph=True, allow_unused=True)[0]
        grad_ux1, grad_ux2 = grad_u[:, 0], grad_u[:, 1]

        physical_term1 = grad_ux1.pow(2) + grad_ux2.pow(2)
        physical_term2 = true_operator_x*output_ux_in_omega
        boundary_loss = self.mse(output_on_boundary, true_value_on_boundary)

        total_loss = 4*0.5*torch.mean(physical_term1) - 4*torch.mean(physical_term2) + boundary_regulizer*boundary_loss
        return total_loss
    
if __name__=="__main__":
    device = torch.device("cuda")
    ritz_model = rnn.RitzModel(2).to(device)
    criterion = DeepRitzLoss()
    model_optimizer = torch.optim.Adam(ritz_model.parameters(), lr=0.0005)

    n_iter_adam = 20000
    n_omega = 100
    n_boundary = 75
    beta = 500

    if isinstance(ritz_model, rnn.RitzModel):
        sig = 'res'
    else:
        sig = 'fwd'
    model_name = f'ex2_{sig}_iter{n_iter_adam//1000}k_{n_omega}_{n_boundary}_beta{beta}.pth'

    ritz_model.train()
    for iter_i in range(n_iter_adam):
        integration_grid = ex2_utils.Corner_Singularity_2D(n_omega, n_boundary, normal=False)
        mc_integration_points_in_omega = torch.FloatTensor(integration_grid.omega_coordinates).to(device)
        mc_integration_points_on_boundary = torch.FloatTensor(integration_grid.boundary_coordinates).to(device)
        mc_true_boundary = torch.FloatTensor(ex2_utils.u_true(mc_integration_points_on_boundary.cpu())).reshape(integration_grid.nb_points_on_boundary, 1).to(device)

        mc_integration_points_in_omega.requires_grad = True

        fmc_integration_points = torch.FloatTensor(ex2_utils.f_true(mc_integration_points_in_omega)).reshape(integration_grid.nb_points_in_omega, 1).to(device)

        model_optimizer.zero_grad()
        mc_output_omega = ritz_model(mc_integration_points_in_omega)
        mc_output_boundary = ritz_model(mc_integration_points_on_boundary)
    
        total_loss = criterion(x_in_omega=mc_integration_points_in_omega, output_ux_in_omega=mc_output_omega, true_operator_x=fmc_integration_points, output_on_boundary=mc_output_boundary, true_value_on_boundary=mc_true_boundary, boundary_regulizer=beta).to(device)

        total_loss.backward()
        model_optimizer.step()

        if iter_i % 100 == 0:
            print(f"Iteration {iter_i} - Loss: {total_loss}")

    torch.save(ritz_model.state_dict(), f'./Ex2_singularity2D/Models/{model_name}')