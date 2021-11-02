import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


def plot_clulsters(X, y):
    plt.figure(figsize=(6, 4))

    # Create a minimum and maximum range of X.
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    # Get the average distance for X.
    X = (X - x_min) / (x_max - x_min)

    # This loop displays all of the datapoints.
    for i in range(X.shape[0]):
        # Replace the data points with their respective cluster value
        # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # Remove the x ticks, y ticks, x and y axis
    plt.xticks([])
    plt.yticks([])
    # plt.axis('off')

    # Display the plot of the original data before clustering
    plt.scatter(X[:, 0], X[:, 1], marker='.')
    # Display the plot
    plt.show()


if __name__ == "__main__":
    X, y = make_blobs(n_samples=50, centers=[[4, 4], [-2, -1], [1, 1], [10, 4]], cluster_std=0.9)
    agglom = AgglomerativeClustering(n_clusters=4, linkage='average')
    agglom.fit(X, y)

    plot_clulsters(X, y)

    dist_matrix = distance_matrix(X, X)
    Z = hierarchy.linkage(dist_matrix, 'complete')
    dendro = hierarchy.dendrogram(Z)
    plt.show()