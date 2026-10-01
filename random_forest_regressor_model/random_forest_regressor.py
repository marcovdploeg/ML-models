# The Python file making a class for the random forest regressor model
# as described in the notebook.

import numpy as np
import pandas as pd
from decision_tree_regressor_modified import DecisionTreeRegressorModified

class RandomForestRegressor:
    def __init__(
        self, 
        n_estimators: int, 
        random_state: int | None = 42, 
        max_depth: int | None = 3,
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1, 
        verbose: bool = False,
        threshold: float = 0.0, 
        num_features: int | None = None
    ) -> None:
        """Initialize the Random Forest Regressor.
        Note: the first two arguments are for the forest model,
        the rest control the subsequent decision trees.

        Args:
            n_estimators (int): Number of trees in the forest.
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
            num_features (int | None): Number of features to consider in
                each node. If None, one-third of the total number of
                features rounded down is used. Defaults to None.
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
    
    def create_bootstrap_samples(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> list:
        """Create the bootstrap samples of the given data.
        
        Args:
            X (pd.DataFrame): The original features dataset.
            y (pd.Series): The target values corresponding to the dataset.
        
        Returns:
            list: The bootstrap samples of the data.
        """
        sample_size = X.shape[0]
        bootstrap_samples = []

        for _ in range(self.n_estimators):
            # Generate a random sample with replacement
            sample_indices = np.random.choice(
                sample_size, size=sample_size, replace=True
            )
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            bootstrap_samples.append( (X_sample, y_sample) )
        return bootstrap_samples
    
    def build_forest(
        self, 
        bootstrap_samples: list
    ) -> list:
        """Build the random forest from a decision tree for
        each bootstrap sample.
        
        Args:
            bootstrap_samples (list): The list of bootstrap samples.
        
        Returns:
            list: The random forest (list of decision trees).
        """
        forest = []
        for i in range(self.n_estimators):
            X_sample, y_sample = bootstrap_samples[i]
            
            # Train a decision tree on the bootstrap sample
            tree = DecisionTreeRegressorModified(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                verbose=self.verbose,
                threshold=self.threshold,
                num_features=self.num_features,
                random_state=self.random_state
            )
            tree.fit(X_sample, y_sample)
            forest.append(tree)
        return forest
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        """Fit the random forest model to the training data.
        
        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target values.
        """
        bootstrap_samples = self.create_bootstrap_samples(X, y)
        self.forest = self.build_forest(bootstrap_samples)
    
    def predict(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make predictions using the random forest.
        
        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted target values for the input data.
        """
        predictions = np.array([tree.predict(X) for tree in self.forest])
        # Use the average
        forest_predictions = np.mean(predictions, axis=0)
        return forest_predictions
    
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit the model
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    rmse = random_forest.score(y_test, y_pred)
    print(f"Root Mean Square Error on iris data: {rmse:.4f}")