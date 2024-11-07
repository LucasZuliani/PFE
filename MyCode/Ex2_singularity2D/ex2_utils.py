import numpy as np
import torch
from torch.utils.data import Dataset
from fenics import RectangleMesh, Point
import matplotlib.pyplot as plt

## Domain definition ##
class Corner_Singularity_2D(Dataset):
    def __init__(self, nb_points_in_omega=1000, nb_points_on_boundary=500, normal=False, lower_bound_xy=[-1, -1], uper_bound_xy=[1, 1], grid_size=19):

        self.lower_bound_xy = lower_bound_xy
        self.upper_bound_xy = uper_bound_xy
        self.squared_grid_size = grid_size
        self.normal = normal

        self.nb_points_in_omega = nb_points_in_omega
        self.nb_points_on_boundary = nb_points_on_boundary
        self.omega_coordinates, self.boundary_coordinates = self._create_points_in_domain(self.nb_points_in_omega, self.nb_points_on_boundary, self.lower_bound_xy, self.upper_bound_xy, self.normal)

        self.uniform_all_points = self._create_uniform_grid([self.squared_grid_size, self.squared_grid_size], self.lower_bound_xy, self.upper_bound_xy)

    def __getitem__(self, index):
        return self.omega_coordinates[index]
    
    def __len__(self):
        return self.nb_points_in_omega + self.nb_points_on_boundary

    def  _create_points_in_domain(self, nb_points_in_omega, nb_points_on_boundary, lower_bound_xy, upper_bound_xy, normal, tol_boundary_corner=1e-6, tol_boundary_square=1e-6):
        xmin, ymin = lower_bound_xy
        xmax, ymax = upper_bound_xy
        if normal:
            x_points_in_omega = np.random.normal(0, 0.66, nb_points_in_omega)
            y_points_in_omega = np.random.normal(0, 0.66, nb_points_in_omega)
            omega_coordinates = np.vstack((x_points_in_omega, y_points_in_omega)).T
            omega_coordinates = omega_coordinates[(omega_coordinates[:, 0] >= xmin) & (omega_coordinates[:, 0] <= xmax)
                                                  & (omega_coordinates[:, 1] >= ymin) & (omega_coordinates[:, 1] <= ymax)]
        else:
            x_points_in_omega = np.random.uniform(xmin+tol_boundary_square, xmax-tol_boundary_square, nb_points_in_omega)
            y_points_in_omega = np.random.uniform(ymin+tol_boundary_square, ymax-tol_boundary_square, nb_points_in_omega)
            omega_coordinates = np.vstack((x_points_in_omega, y_points_in_omega)).T

        corner_singularity_mask = ~((-tol_boundary_corner <= omega_coordinates[:, 1]) &
                                    (omega_coordinates[:, 1] <= tol_boundary_corner) &
                                    (0 <= omega_coordinates[:, 0]) &
                                    (omega_coordinates[:, 0] <= xmax-tol_boundary_square))
        omega_coordinates = omega_coordinates[corner_singularity_mask]

        nb_points_per_boundary = nb_points_on_boundary // 5
        x_bottom, y_bottom = np.random.uniform(xmin, xmax, nb_points_per_boundary), np.ones(nb_points_per_boundary)*ymin
        x_top, y_top = np.random.uniform(xmin, xmax, nb_points_per_boundary), np.ones(nb_points_per_boundary)*ymax
        x_right, y_right = np.ones(nb_points_per_boundary)*xmax, np.random.uniform(ymin, ymax, nb_points_per_boundary)
        x_left, y_left = np.ones(nb_points_per_boundary)*xmin, np.random.uniform(ymin, ymax, nb_points_per_boundary)
        x_corner, y_corner = np.random.uniform(0, xmax, nb_points_per_boundary), np.zeros(nb_points_per_boundary)

        x_boundary = np.concatenate((x_bottom, x_top, x_right, x_left, x_corner))
        y_boundary = np.concatenate((y_bottom, y_top, y_right, y_left, y_corner))

        boundary_coordinates = np.vstack((x_boundary, y_boundary)).T

        self.nb_points_on_boundary = len(boundary_coordinates)
        self.nb_points_in_omega = len(omega_coordinates)

        return omega_coordinates, boundary_coordinates

    def _create_uniform_grid(self, grid_size, lower_bound_xy, upper_bound_xy):
        mesh = RectangleMesh(Point(lower_bound_xy[0], lower_bound_xy[1]), Point(upper_bound_xy[0], upper_bound_xy[1]), grid_size[0], grid_size[1])
        mesh_coordinates = mesh.coordinates()

        return mesh_coordinates

    def plot_domain(self, label=True):
        plt.scatter(self.omega_coordinates[:, 0], self.omega_coordinates[:, 1], c='blue', alpha=0.6, label='Omega points')
        plt.scatter(self.boundary_coordinates[:, 0], self.boundary_coordinates[:, 1], c='red', alpha=0.6, label='Boundary points')
        plt.title('Domain with corner singularity along the x-axis at y=0')
        plt.xlabel('x')
        plt.ylabel('y')
        if label:
            plt.legend()
        plt.show()   

## Relative to exact solution ##

def cart2pol(x, y):
    if isinstance(x, np.ndarray):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        negative_mask = phi < 0
        phi[negative_mask] += 2 * np.pi
    else:
        rho = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)
        negative_mask = phi < 0
        phi[negative_mask] += torch.FloatTensor([2 * np.pi]).to(x.device)
    return(rho, phi)

def u_true(x):
    x, y = x[:, 0], x[:, 1]
    r, theta = cart2pol(x, y)
    if isinstance(x, np.ndarray):
        return r**(1/2) * np.sin(theta/2)
    else:
        return r**(1/2) * torch.sin(theta/2)

def du_x1_true(x):
    x, y = x[:, 0], x[:, 1]
    r, theta = cart2pol(x, y)

    df_dr = torch.sin(theta/2) / (2*torch.sqrt(r))
    dr_dx = x / r
    c1 = df_dr * dr_dx

    df_dtheta = torch.sqrt(r) * torch.cos(theta/2) / 2
    dtheta_dx = -y / (x**2 + y**2)
    c2 = df_dtheta * dtheta_dx

    return c1+c2

def du_x2_true(x):
    x, y = x[:, 0], x[:, 1]
    r, theta = cart2pol(x, y)

    df_dr = torch.sin(theta/2) / (2*torch.sqrt(r))
    dr_dy = y / r
    c1 = df_dr * dr_dy

    df_dtheta = torch.sqrt(r) * torch.cos(theta/2) / 2
    dtheta_dy = x / (x**2 + y**2)
    c2 = df_dtheta * dtheta_dy

    return c1+c2

def f_true(x):
    return np.zeros(x.shape[0])

if __name__ == "__main__":
    domain = Corner_Singularity_2D(nb_points_in_omega=1000, nb_points_on_boundary=500, normal=False)
    domain.plot_domain()