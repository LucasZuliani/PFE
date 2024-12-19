import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch 
from torch.utils.data import Dataset
from fenics import *

import plotly.graph_objects as go

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from NeuralNetworks import fwdNN as fwd

class DeepRitzLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x_omega, ux_omega, fx, ux_boundary, cl_boundary, reg_boundary=500):
        grad_u = torch.autograd.grad(inputs=x_omega, outputs=ux_omega, grad_outputs=torch.ones_like(ux_omega), create_graph=True)[0]
        grad_ux1, grad_ux2, grad_ux3 = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
        
        physical_term1 = grad_ux1.pow(2) + grad_ux2.pow(2) + grad_ux3.pow(2)
        physical_term2 = fx*ux_omega
        boundary_term = (cl_boundary - ux_boundary).pow(2)

        vol = 8
        loss_term1 = vol/2*physical_term1.mean()
        loss_term2 = vol*physical_term2.mean()
        loss_term3 = boundary_term.mean()
        loss = loss_term1 - loss_term2 + reg_boundary*loss_term3

        return loss, loss_term1, loss_term2, loss_term3
    
class Omega3D(Dataset):
    def __init__(self, n_omega=100, n_boundary=75):
        self.n_omega = n_omega
        self.n_boundary = n_boundary

        self.omega_train, self.boundary_train, self.integration_points = self._create_points_for_training()

    def _create_points_for_training(self):
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        zmin, zmax = -1, 1
        n = self.n_omega
        n_per_face = self.n_boundary // 6

        x, y, z = np.random.uniform(-1, 1, n), np.random.uniform(-1, 1, n), np.random.uniform(-1, 1, n)
        omega_train = np.column_stack((x, y, z))


        x_left, y_left, z_left = xmin * np.ones(n_per_face), np.random.uniform(ymin, ymax, n_per_face), np.random.uniform(zmin, zmax, n_per_face)
        x_right, y_right, z_right = xmax * np.ones(n_per_face), np.random.uniform(ymin, ymax, n_per_face), np.random.uniform(zmin, zmax, n_per_face)
        x_front, y_front, z_front = np.random.uniform(xmin, xmax, n_per_face), ymin * np.ones(n_per_face), np.random.uniform(zmin, zmax, n_per_face)
        x_back, y_back, z_back = np.random.uniform(xmin, xmax, n_per_face), ymax * np.ones(n_per_face), np.random.uniform(zmin, zmax, n_per_face)
        x_bottom, y_bottom, z_bottom = np.random.uniform(xmin, xmax, n_per_face), np.random.uniform(ymin, ymax, n_per_face), zmin * np.ones(n_per_face)
        x_top, y_top, z_top = np.random.uniform(xmin, xmax, n_per_face), np.random.uniform(ymin, ymax, n_per_face), zmax * np.ones(n_per_face)

        x_boundary = np.concatenate((x_left, x_right, x_front, x_back, x_bottom, x_top))
        y_boundary = np.concatenate((y_left, y_right, y_front, y_back, y_bottom, y_top))
        z_boundary = np.concatenate((z_left, z_right, z_front, z_back, z_bottom, z_top))
        boundary_train = np.column_stack((x_boundary, y_boundary, z_boundary))

        integration_points = np.concatenate((omega_train, boundary_train))

        return omega_train, boundary_train, integration_points
    
    def plot_domain(self):
        xmin, xmax = -1, 1
        ymin, ymax = -1, 1
        zmin, zmax = -1, 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        vertices = [
            [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin]],  # face inférieure
            [[xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax]],  # face supérieure
            [[xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymin, zmax], [xmin, ymin, zmax]],  # face avant
            [[xmin, ymax, zmin], [xmax, ymax, zmin], [xmax, ymax, zmax], [xmin, ymax, zmax]],  # face arrière
            [[xmin, ymin, zmin], [xmin, ymax, zmin], [xmin, ymax, zmax], [xmin, ymin, zmax]],  # face gauche
            [[xmax, ymin, zmin], [xmax, ymax, zmin], [xmax, ymax, zmax], [xmax, ymin, zmax]],  # face droite
        ]

        face_colors = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray']
        for i, verts in enumerate(vertices):
            ax.add_collection3d(Poly3DCollection([verts], color=face_colors[i], linewidths=1, edgecolors='k', alpha=0.2))

        ax.scatter(self.omega_train[:, 0], self.omega_train[:, 1], self.omega_train[:, 2], c='b', marker='o', label='Omega')
        ax.scatter(self.boundary_train[:, 0], self.boundary_train[:, 1], self.boundary_train[:, 2], c='r', marker='o', label='Boundary')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.show()

        return

class Circle():
    def __init__(self, hidden_size=20, nn='res'):

        if nn == 'fwd':
            n_hidden_layers = int(input('Number of hidden layers: '))
            self.model = fwd.FullyConnectedNetwork(input_dim=3, n_hidden_layers=n_hidden_layers, hidden_size=hidden_size).to(device)

        self.criterion = DeepRitzLoss()

    def train(self, n_iter, n_omega, n_boundary, beta):
        
        criterion = self.criterion
        model_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        self.model.train()

        for iter_i in range(n_iter):
            integration_grid = Omega3D(n_omega=n_omega, n_boundary=n_boundary)

            x_omega = torch.FloatTensor(integration_grid.omega_train).to(device)
            x_boundary = torch.FloatTensor(integration_grid.boundary_train).to(device)
            fx_omega = torch.FloatTensor(self.f(x_omega.cpu()).unsqueeze(1)).to(device)
            cl_boundary = torch.FloatTensor(self.cl_boundary(x_boundary)).to(device)
            x_omega.requires_grad = True

            model_optimizer.zero_grad()
            ux_omega = self.model(x_omega)
            ux_boundary = self.model(x_boundary)
            loss, physical_loss1, physical_loss2, boundary_loss = criterion(x_omega, ux_omega, fx_omega, ux_boundary, cl_boundary, beta)

            loss.backward()
            model_optimizer.step()

            if iter_i % 100 == 0:
                print(f'Iteration {iter_i}, loss: {loss.item()}')

    def f(self, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        fu = 2*( (1-x2**2)*(1-x3**2) + (1-x1**2)*(1-x3**2) + (1-x1**2)*(1-x2**2) )
        return fu
    
    def u(self, x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        u = (1-x1**2)*(1-x2**2)*(1-x3**2)
        return u
    
    def cl_boundary(self, x):
        return torch.zeros(x.shape[0], 1)
    
    def plot_metrics(self, evaluation_grid_size):
        mesh_coordinates = (BoxMesh(Point(-1, -1, -1), Point(1, 1, 1), evaluation_grid_size, evaluation_grid_size, evaluation_grid_size)).coordinates()
        evaluation_domain_points = torch.FloatTensor(mesh_coordinates).to(device)
        evaluation_domain_points.requires_grad = True

        model = self.model
        model.eval()

        u_pred = model(evaluation_domain_points)
        u_exact = (self.u(evaluation_domain_points.cpu()).unsqueeze(1)).to(device)

        fig1 = go.Figure(data=go.Isosurface(
            x=mesh_coordinates[:, 0], y=mesh_coordinates[:, 1], z=mesh_coordinates[:, 2],
            value=u_exact.cpu().detach().numpy().ravel(),  # Les valeurs scalaires
            isomin=u_exact.min().item(),  # Valeur minimale pour l'isovolume
            isomax=u_exact.max().item(),  # Valeur maximale pour l'isovolume
            surface_count=30,  # Nombre d'isosurfaces
            colorscale='Jet',
            opacity=0.8,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig1.update_layout(scene=dict(aspectmode='cube'))
        fig1.show()

        fig2 = go.Figure(data=go.Isosurface(
            x=mesh_coordinates[:, 0], y=mesh_coordinates[:, 1], z=mesh_coordinates[:, 2],
            value=u_pred.cpu().detach().numpy().ravel(),  # Les valeurs scalaires
            isomin=u_pred.min().item(),  # Valeur minimale pour l'isovolume
            isomax=u_pred.max().item(),  # Valeur maximale pour l'isovolume
            surface_count=30,  # Nombre d'isosurfaces
            colorscale='Jet',
            opacity=0.8,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig2.update_layout(scene=dict(aspectmode='cube'))
        fig2.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # domain = Omega3D(n_omega=100, n_boundary=75)
    # domain.plot_domain()
    print('')
    model = Circle(hidden_size=20, nn='fwd')
    model.train(15000, 10000, 3000, 4000)
    model.plot_metrics(19)