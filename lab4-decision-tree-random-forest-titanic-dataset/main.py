import numpy as np

from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import generate_data, load_titanic


def main():
    np.random.seed(4289)

    train_data, test_data = load_titanic()

    print("Decision tree")
    dt = DecisionTree({"depth": 16})
    # dt = DecisionTree({"depth": 4})
    dt.train(*train_data)

    print("Train data")
    dt.evaluate(*train_data)
    print("Test data")
    dt.evaluate(*test_data)

    print("\nRandom forest")
    # rf = RandomForest({"ntrees": 10, "feature_subset": 5, "depth": 24})
    # rf = RandomForest({"ntrees": 30, "feature_subset": 5, "depth": 24})
    # rf = RandomForest({"ntrees": 50, "feature_subset": 5, "depth": 30})
    rf = RandomForest({"ntrees": 50, "feature_subset": 6, "depth": 30})
    # rf = RandomForest({"ntrees": 3, "feature_subset": 4, "depth": 24})
    rf.train(*train_data)

    print("Train data")
    rf.evaluate(*train_data)
    print("Test data")
    rf.evaluate(*test_data)


if __name__ == "__main__":
    main()
