# The Python file making a class for the gradient boosting regressor 
# model thought up in the notebook.

import numpy as np
import pandas as pd

# Add the parent directory to Python's path to import the decision tree
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from decision_tree_regressor_model.decision_tree_regressor import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators, learning_rate, subsample, random_state=42, 
                 max_depth=3, min_samples_split=2, min_samples_leaf=1,
                 verbose=False, threshold=0.0):
        """
        Initialize the Gradient Boosting Regressor, 
        using the MSE loss function.

        Parameters:
        n_estimators (int): The number of trees to fit.
        learning_rate (float): The learning rate for updating predictions.
        subsample (float): The fraction of the dataset to be used for 
                           fitting each tree.
        random_state (int, optional): Random seed for reproducibility.
                                      Default is 42.

        Parameters (decision tree):
        max_depth (int, optional): Maximum depth of the tree.
                                   Default is 3.
        min_samples_split (int, optional): Minimum number of samples required
                                     to split an internal node. Default is 2.
        min_samples_leaf (int, optional): Minimum number of samples required
                                         to be at a leaf node. Default is 1.
        verbose (bool, optional): If True, print the reasons for stopping.
                                  Default is False.
        threshold (float, optional): The threshold to compare against
                                     for increasing score. Default is 0.0.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.threshold = threshold
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.y_pred_initial = None  # to hold the initial prediction
        self.tree_chain = []  # to hold the trees after training
    
    def pseudo_residuals(self, y_true, y_pred):
        """
        Calculate the pseudo-residuals for the MSE loss
        that the decision tree will be fit to, which are
        the actual residuals in this case.

        Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

        Returns:
        array-like: Pseudo-residuals.
        """
        return (y_true - y_pred)

    def initial_prediction(self, y_true):
        """
        Calculate the initial prediction for the gradient 
        boosting algorithm, which is the mean of the 
        target values for the MSE loss.

        Parameters:
        y_true (array-like): True values.

        Returns:
        float: Initial prediction.
        """
        return np.mean(y_true)
    
    def create_bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the given data.
        
        Parameters:
        X (DataFrame): The original dataset.
        y (Series): The labels corresponding to the dataset.
        
        Returns:
        X_sample (DataFrame): The bootstrap sample of the dataset.
        y_sample (Series): The labels corresponding to the bootstrap sample.
        """
        sample_size = int(self.subsample * len(X))

        # Generate a random sample without replacement
        sample_indices = np.random.choice(sample_size, size=sample_size, 
                                          replace=False)
        X_sample, y_sample = X.iloc[sample_indices], y.iloc[sample_indices]
        return X_sample, y_sample
    
    def fit(self, X, y):
        """
        Build a gradient boosting model by fitting a chain of 
        decision trees to the pseudo-residuals.

        Parameters:
        X (DataFrame): The input features.
        y (Series): The target values.
        """
        self.y_pred_initial = self.initial_prediction(y)
        
        y_pred_for_residuals = self.y_pred_initial.copy()
        for _ in range(self.n_estimators):
            pseudo_residuals_i = self.pseudo_residuals(y, y_pred_for_residuals)
            X_sample, y_sample = self.create_bootstrap_sample(X, pseudo_residuals_i)

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        verbose=self.verbose,
                                        threshold=self.threshold)
            tree.fit(X_sample, y_sample)
            self.tree_chain.append(tree)

            # Add current predictions for the residuals in next loop
            y_pred_for_residuals += self.learning_rate * tree.predict(X)

    def predict(self, X):
        """
        Make predictions using the fitted gradient boosting model.

        Parameters:
        X (DataFrame): The input features for which to make predictions.

        Returns:
        array-like: The predicted values for the input features.
        """
        y_pred = self.y_pred_initial.copy()
        for tree in self.tree_chain:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

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
    print("Testing the GradientBoostingRegressor class.")

    gradient_booster = GradientBoostingRegressor(n_estimators=30, 
                                                 learning_rate=0.1, 
                                                 subsample=1.0)
    
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
    gradient_booster.fit(X_train, y_train)
    y_pred = gradient_booster.predict(X_test)
    rmse = gradient_booster.score(y_test, y_pred)
    print(f"Root Mean Square Error on iris data: {rmse:.4f}")