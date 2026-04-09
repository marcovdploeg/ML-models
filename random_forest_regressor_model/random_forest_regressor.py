# The Python file making a class for the random forest regressor model
# as described in the notebook.

import numpy as np
import pandas as pd
from decision_tree_regressor_modified import DecisionTreeRegressorModified

class RandomForestRegressor:
    def __init__(self, n_estimators, random_state=42, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, verbose=False,
                 threshold=0.0, num_features=None):
        """
        Initialize the Random Forest Classifier.

        Parameters:
        n_estimators (int): Number of trees in the forest.
        random_state (int, optional): Random seed for reproducibility.
                                      Default is 42.

        Parameters (decision tree):
        max_depth (int, optional): Maximum depth of the tree.
                                   Default is None (no limit).
        min_samples_split (int, optional): Minimum number of samples required
                                     to split an internal node. Default is 2.
        min_samples_leaf (int, optional): Minimum number of samples required
                                         to be at a leaf node. Default is 1.
        verbose (bool, optional): If True, print the reasons for stopping.
                                  Default is False.
        threshold (float, optional): The threshold to compare against
                                     for increasing score. Default is 0.0.
        num_features (int, optional): Number of features to consider in each
                                      node. If None, one-third of the 
                                      total number of features rounded down
                                      is used. Default is None.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.threshold = threshold
        self.num_features = num_features
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.forest = None  # to hold the forest after training trees
    
    def create_bootstrap_samples(self, X, y):
        """
        Create the bootstrap samples of the given data.
        
        Parameters:
        X (DataFrame): The original dataset.
        y (Series): The target values corresponding to the dataset.
        
        Returns:
        list: The bootstrap samples of the data.
        """
        sample_size = X.shape[0]
        bootstrap_samples = []

        for _ in range(self.n_estimators):
            # Generate a random sample with replacement
            sample_indices = np.random.choice(sample_size,
                                              size=sample_size,
                                              replace=True)
            X_sample, y_sample = X.iloc[sample_indices], y.iloc[sample_indices]
            bootstrap_samples.append( (X_sample, y_sample) )
        
        return bootstrap_samples
    
    def build_forest(self, bootstrap_samples):
        """
        Build the random forest from a decision tree for each bootstrap sample.
        
        Parameters:
        bootstrap_samples (list): The list of bootstrap samples.
        
        Returns:
        list: The random forest (list of decision trees).
        """
        forest = []
        
        for i in range(self.n_estimators):
            X_sample, y_sample = bootstrap_samples[i]
            
            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeRegressorModified(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split,
                                    min_samples_leaf=self.min_samples_leaf,
                                    verbose=self.verbose,
                                    threshold=self.threshold,
                                    num_features=self.num_features,
                                    random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            forest.append(tree)
        
        return forest
    
    def fit(self, X, y):
        """
        Fit the random forest model to the training data.
        
        Parameters:
        X (DataFrame): Feature dataframe.
        y (Series): Target values.
        """
        bootstrap_samples = self.create_bootstrap_samples(X, y)
        self.forest = self.build_forest(bootstrap_samples)
    
    def predict(self, X):
        """
        Make predictions using the random forest.
        
        Parameters:
        X (DataFrame): The data to make predictions on.

        Returns:
        array-like: The predicted target values for the input data.
        """
        predictions = np.array([tree.predict(X) for tree in self.forest])
        # Use the average
        forest_predictions = np.mean(predictions, axis=0)
        return forest_predictions
    
    def score(self, y_true, y_pred):
        """
        Calculate the root mean square error of predictions.
        
        Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        
        Returns:
        float: Root mean square error.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

if __name__ == "__main__":
    print("Testing the RandomForestRegressor class.")

    random_forest = RandomForestRegressor(n_estimators=5, max_depth=5)
    iris_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    iris_data = pd.read_csv(iris_url, header=None)
    X = iris_data.drop(columns=[0])
    y = iris_data[0]

    # Map the labels to integers
    label_mapping = {label: idx for idx, label in enumerate(X[4].unique())}
    X[4] = X[4].map(label_mapping)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    rmse = random_forest.score(y_test, y_pred)
    print(f"Root Mean Square Error on iris data: {rmse:.4f}")