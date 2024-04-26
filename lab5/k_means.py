import numpy as np


def initialize_centroids_forgy(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]


def initialize_centroids_kmeans_pp(data, k):
    centroids = [data[np.random.randint(data.shape[0])]]
    for _ in range(1, k):
        distances = np.array([min([np.inner(cent-data_point, cent-data_point) for cent in centroids]) for data_point in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = probabilities.cumsum()
        r = np.random.rand()
        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[i])
                break
    return np.array(centroids)


def assign_to_cluster(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(data, assignments, num_centroids):
    return np.array([data[assignments == k].mean(axis=0) if np.any(assignments == k) else data[np.random.randint(len(data))] for k in range(num_centroids)])


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initialization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)
    for i in range(100):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments, num_centroids)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return assignments, centroids, mean_intra_distance(data, assignments, centroids)
