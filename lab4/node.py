import copy

import numpy as np


def gini_impurity(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    gini = 1 - np.sum(np.square(probabilities))
    return gini


def gini_best_score(y, possible_splits):
    best_gain = -np.inf
    best_idx = 0
    initial_impurity = gini_impurity(y)

    for idx in possible_splits:
        left_y = y[:idx + 1]
        right_y = y[idx + 1:]
        left_impurity = gini_impurity(left_y)
        right_impurity = gini_impurity(right_y)
        n = len(y)
        weighted_impurity = (len(left_y) / n) * left_impurity + (len(right_y) / n) * right_impurity
        gain = initial_impurity - weighted_impurity

        if gain > best_gain:
            best_gain = gain
            best_idx = idx

    return best_idx, best_gain


def split_data(X, y, idx, val):
    left_mask = X[:, idx] < val
    return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])


def find_possible_splits(data):
    possible_split_points = []
    for idx in range(data.shape[0] - 1):
        if data[idx] != data[idx + 1]:
            possible_split_points.append(idx)
    return possible_split_points


def find_best_split(X, y, feature_subset=None):
    best_gain = -np.inf
    best_split = None
    n_features = X.shape[1]

    # IMPORTANT: select a subset of features at random if feature_subset is specified
    if feature_subset is not None and feature_subset < n_features:
        features = np.random.choice(n_features, feature_subset, replace=False)
    else:
        features = range(n_features)  # IMPORTANT: use all features if no subset is specified

    for d in features:
        order = np.argsort(X[:, d])
        X_sorted = X[order]
        y_sorted = y[order]
        possible_splits = find_possible_splits(X_sorted[:, d])

        idx, gain = gini_best_score(y_sorted, possible_splits)
        if gain > best_gain:
            best_gain = gain
            best_split = (d, idx)

    if best_split is None:
        return None, None

    # IMPORTANT: calculate the value to split at by finding the mean of the values at the best index and its next element
    split_feature, split_index = best_split
    if split_index < len(X) - 1:
        best_value = (X_sorted[split_index, split_feature] + X_sorted[split_index + 1, split_feature]) / 2
    else:
        best_value = X_sorted[split_index, split_feature]

    return split_feature, best_value


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
