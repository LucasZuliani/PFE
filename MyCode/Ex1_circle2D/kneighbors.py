from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import argparse

def kneighbors(domain, k, metric='euclidean'):
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(domain)
    return nn

def recover_neighbors(nn, query_point):
    distances, indices = nn.kneighbors(query_point)
    return distances, indices

def intermediate_neighbors(query_points, neighbors_points):
    return (query_points + neighbors_points) / 2

def distance_threshold_filter(points, distances, indices, threshold):
    valid_indices = indices[distances <= threshold]
    return points[valid_indices]

def plot_neighbors_2D(points, query_point, indices, k=5):
    intermediate_points = intermediate_neighbors(query_point, points[indices])
    neighbors = points[indices]

    assert intermediate_points.shape == (1, k, 2)
    assert neighbors.shape == (1, k, 2)

    plt.scatter(points[:, 0], points[:, 1], color='black')
    plt.scatter(query_point[:, 0], query_point[:, 1], color='red')
    plt.scatter(neighbors[0, :, 0], neighbors[0, :, 1], color='blue')
    plt.scatter(intermediate_points[0, :, 0], intermediate_points[0, :, 1], color='orange')

    for i in indices[0]:  # Parcourir les indices des voisins
        plt.plot([query_point[0, 0], points[i, 0]], 
                 [query_point[0, 1], points[i, 1]], 
                 color='green', linestyle='--')
        
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--example', type=bool, default=False)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    if args.example:
        points = np.random.rand(50, 2) * 10
        query_point = np.array([[4, 4]])
        k = args.k

        nn = kneighbors(points, k)
        distances, indices = recover_neighbors(nn, query_point)

        plot_neighbors_2D(points, query_point, indices, k)

