# The Python file making a class for the gradient boosting regressor
# model thought up in the notebook.

import numpy as np
import pandas as pd

# Add the parent directory to Python's path to import the decision tree
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from decision_tree_regressor_model.decision_tree_regressor import (
    DecisionTreeRegressor
)

class GradientBoostingRegressor:
    def __init__(
        self, 
        n_estimators: int, 
        learning_rate: float, 
        subsample: float, 
        random_state: int | None = 42, 
        max_depth: int | None = 3, 
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1,
        verbose: bool = False, 
        threshold: float = 0.0
    ) -> None:
        """Initialize the Gradient Boosting Regressor, 
        using the MSE loss function.
        Note: the first four arguments are for the gradient boosting model,
        the rest control the subsequent decision trees.

        Args:
            n_estimators (int): The number of trees to fit.
            learning_rate (float): The learning rate for updating predictions.
            subsample (float): The fraction of the dataset to be used for
                fitting each tree.
            random_state (int | None): Random seed for reproducibility.
                Omitted if None. Defaults to 42.
            max_depth (int | None): Maximum depth of the tree. If None,
                there is no limit. Defaults to 3.
            min_samples_split (int): Minimum number of samples required
                to split an internal node. Defaults to 2.
            min_samples_leaf (int): Minimum number of samples required
                to be at a leaf node. Defaults to 1.
            verbose (bool): If True, print the reasons for stopping.
                Defaults to False.
            threshold (float): The threshold to compare against for
                increasing score. Defaults to 0.0.
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
    
    def pseudo_residuals(
        self, 
        y_true: np.ndarray | pd.Series, 
        y_pred: np.ndarray | pd.Series
    ) -> np.ndarray | pd.Series:
        """Calculate the pseudo-residuals for the MSE loss
        that the decision tree will be fit to, which are
        the actual residuals in this case.

        Args:
            y_true (np.ndarray | pd.Series): True values.
            y_pred (np.ndarray | pd.Series): Predicted values.

        Returns:
            np.ndarray | pd.Series: Pseudo-residuals.
        """
        return (y_true - y_pred)

    def initial_prediction(
        self, 
        y_true: np.ndarray | pd.Series
    ) -> float:
        """Calculate the initial prediction for the gradient
        boosting algorithm, which is the mean of the
        target values for the MSE loss.

        Args:
            y_true (np.ndarray | pd.Series): True values.

        Returns:
            float: Initial prediction.
        """
        return np.mean(y_true)
    
    def create_bootstrap_sample(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create a bootstrap sample of the given data.
        
        Args:
            X (pd.DataFrame): The original features dataset.
            y (pd.Series): The target values corresponding to the dataset.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X_sample (pd.DataFrame): The bootstrap sample of
                    the dataset.
                - y_sample (pd.Series): The target values corresponding to
                    the bootstrap sample.
        """
        sample_size = int(self.subsample * len(X))

        # Generate a random sample without replacement
        sample_indices = np.random.choice(
            sample_size, size=sample_size, replace=False
        )
        X_sample, y_sample = X.iloc[sample_indices], y.iloc[sample_indices]
        return X_sample, y_sample
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        """Build a gradient boosting model by fitting a chain of
        decision trees to the pseudo-residuals.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target values.
        """
        self.y_pred_initial = self.initial_prediction(y)
        
        y_pred_for_residuals = self.y_pred_initial.copy()
        for _ in range(self.n_estimators):
            pseudo_residuals_i = self.pseudo_residuals(
                y, y_pred_for_residuals
            )
            X_sample, y_sample = self.create_bootstrap_sample(
                X, pseudo_residuals_i
            )

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                verbose=self.verbose,
                threshold=self.threshold
            )
            tree.fit(X_sample, y_sample)
            self.tree_chain.append(tree)

            # Add current predictions for the residuals in next loop
            y_pred_for_residuals += self.learning_rate * tree.predict(X)

    def predict(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make predictions using the fitted gradient boosting model.

        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted target values for the input data.
        """
        y_pred = self.y_pred_initial.copy()
        for tree in self.tree_chain:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def score(
        self, 
        y_true: np.ndarray | pd.Series, 
        y_pred: np.ndarray | pd.Series
    ) -> float:
        """Calculate the root mean square error of predictions.
        
        Args:
            y_true (np.ndarray | pd.Series): True values.
            y_pred (np.ndarray | pd.Series): Predicted values.
        
        Returns:
            float: Root mean square error.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

if __name__ == "__main__":
    print("Testing the GradientBoostingRegressor class.")

    gradient_booster = GradientBoostingRegressor(
        n_estimators=30,
        learning_rate=0.1,
        subsample=1.0
    )
    iris_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    iris_data = pd.read_csv(iris_url, header=None)
    X = iris_data.drop(columns=[0])
    y = iris_data[0]

    # Map the labels to integers
    label_mapping = {label: idx for idx, label in enumerate(X[4].unique())}
    X[4] = X[4].map(label_mapping)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit the model
    gradient_booster.fit(X_train, y_train)
    y_pred = gradient_booster.predict(X_test)
    rmse = gradient_booster.score(y_test, y_pred)
    print(f"Root Mean Square Error on iris data: {rmse:.4f}")