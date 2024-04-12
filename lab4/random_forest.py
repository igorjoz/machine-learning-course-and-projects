from collections import defaultdict
import numpy as np

from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, params):
        self.forest = []
        self.params = defaultdict(lambda: None, params)

    def train(self, X, y):
        for _ in range(self.params["ntrees"]):
            X_bagging, y_bagging = self.bagging(X, y)
            tree = DecisionTree(self.params)
            tree.train(X_bagging, y_bagging)
            self.forest.append(tree)

    def evaluate(self, X, y):
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted == y), 2)}")

    def predict(self, X):
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(X))
        forest_predictions = list(map(lambda x: sum(x) / len(x), zip(*tree_predictions)))
        return forest_predictions

    def bagging(self, X, y):
        idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_selected, y_selected = X[idx], y[idx]

        return X_selected, y_selected

    # def bagging(self, X, y):
    #     X_selected, y_selected = None, None
    #     # TODO implement bagging
    #
    #     return X_selected, y_selected
    # def bagging(self, X, y):
    #     n_samples = X.shape[0]
    #     indices = np.random.choice(n_samples, size=n_samples, replace=True)
    #     if "feature_subset" in self.params and self.params["feature_subset"] is not None:
    #         n_features = X.shape[1]
    #         features_indices = np.random.choice(n_features, size=self.params["feature_subset"], replace=False)
    #         X_selected = X[indices][:, features_indices]
    #         return X_selected, y[indices]
    #     else:
    #         return X[indices], y[indices]
