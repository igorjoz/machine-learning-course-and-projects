from k_means import k_means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_clusters(data, assignments, centroids):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=assignments, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.6, label='Data points')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    plt.title('Visualization of Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()


def load_iris():
    data = pd.read_csv("data/iris.data", names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    print(data)
    classes = data["class"].to_numpy()
    features = data.drop("class", axis=1).to_numpy()
    return features, classes


def evaluate(clusters, labels):
    for cluster in np.unique(clusters):
        labels_in_cluster = labels[clusters == cluster]
        print(f"Cluster: {cluster}")
        for label_type in np.unique(labels):
            print(f"Num of {label_type}: {np.sum(labels_in_cluster == label_type)}")


def clustering(kmeans_pp):
    data = load_iris()
    features, classes = data
    intra_class_variance = []
    for i in range(1000):
        assignments, centroids, error = k_means(features, 3, kmeans_pp)
        evaluate(assignments, classes)
        intra_class_variance.append(error)
    print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")
    plot_clusters(features, assignments, centroids)


# def clustering(kmeans_pp):
#     data = load_iris()
#     features, classes = data
#     intra_class_variance = []
#     assignments, centroids, error = k_means(features, 3, kmeans_pp)
#     plot_clusters(features, assignments, centroids)  # Add this to plot results after clustering
#     evaluate(assignments, classes)
#     intra_class_variance.append(error)
#     print(f"Mean intra-class variance: {np.mean(intra_class_variance)}")


if __name__ == "__main__":
    clustering(kmeans_pp=True)
    clustering(kmeans_pp=False)
