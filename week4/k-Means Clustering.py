import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def plot_clusters(k_means, X):
    fig = plt.figure(figsize=(6, 4))
    # Colors uses a color map, which will produce an array of colors based on
    # the number of labels there are
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means.labels_))))

    ax = fig.add_subplot(1, 1, 1)
    # For loop that plots the data points and centroids.
    # k will range from 0-3, which will match the possible clusters that each
    # data point is in.
    for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
        # Create a list of all data points, where the data points that are
        # in the cluster (ex. cluster 0) are labeled as true, else they are
        # labeled as false.
        my_members = (k_means.labels_ == k)
        # Define the centroid, or cluster center.
        cluster_center = k_means.cluster_centers_[k]
        # Plots the datapoints with color col.
        ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
        # Plots the centroids with specified color, but with a darker outline
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k',
                markersize=6)

    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    X, y = make_blobs(n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

    plt.scatter(X[:, 0], X[:, 1], marker='.')
    plt.show()

    # Selects initial cluster centers for k - mean clustering in a smart way to speed up convergence
    # n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    # The final results will be the best output of n_init consecutive runs in terms of inertia
    k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
    k_means.fit(X)

    plot_clusters(k_means, X)

