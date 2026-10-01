# The Python file making a class for the random forest classifier model
# as described in the notebook.

import numpy as np
import pandas as pd
from decision_tree_classifier_modified import DecisionTreeClassifierModified

class RandomForestClassifier:
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
        """Initialize the Random Forest Classifier.
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
                each node. If None, the square root of the total number of
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
            y (pd.Series): The labels corresponding to the dataset.
        
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
            tree = DecisionTreeClassifierModified(
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
            y (pd.Series): Labels.
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
            np.ndarray: The predicted labels for the input data.
        """
        predictions = np.array([tree.predict(X) for tree in self.forest])
        # Use majority voting
        forest_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )
        return forest_predictions
    
    def predict_proba(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make predictions for probability distributions
        using the random forest.
        
        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted probability distributions for
                the input data.
        """
        predictions = np.array(
            [tree.predict_proba(X) for tree in self.forest]
        )
        # Average the predictions of all trees
        forest_predictions = np.apply_along_axis(
            lambda x: np.mean(x, axis=0), axis=0, arr=predictions
        )
        return forest_predictions
    
    def score(
        self, 
        y_true: np.ndarray | pd.Series, 
        y_pred: np.ndarray | pd.Series
    ) -> float:
        """Calculate the accuracy of predictions.
        
        Args:
            y_true (np.ndarray | pd.Series): True labels.
            y_pred (np.ndarray | pd.Series): Predicted labels.
        
        Returns:
            float: Accuracy score.
        """
        return np.mean(y_true == y_pred)
    
    def log_loss(
        self, 
        y_true: np.ndarray | pd.Series, 
        y_pred_proba: np.ndarray | pd.Series, 
        epsilon: float = 1e-10
    ) -> float:
        """Calculate the log loss between true labels
        and predicted probabilities.

        Args:
            y_true (np.ndarray | pd.Series): True labels, can be
                one-hot encoded.
            y_pred_proba (np.ndarray | pd.Series): Predicted probabilities for
                each class.
            epsilon (float): Small value to avoid log(0). Defaults to 1e-10.

        Returns:
            float: Log loss value.
        """
        # Clip probabilities to avoid log(0)
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

        # Convert y_true to the same shape as y_pred_proba 
        # if necessary (so if more than 2 classes)
        if y_true.ndim == 1:
            y_true = np.eye(len(y_pred_proba[0]))[y_true]

        # Calculate log loss, with mean for the (1/N) sum
        # Note we don't need to do the second sum over labels, 
        # as y_true is already one-hot encoded
        loss = -np.mean(y_true * np.log(y_pred_proba))
        return loss
    
if __name__ == "__main__":
    print("Testing the RandomForestClassifier class.")
    
    random_forest = RandomForestClassifier(n_estimators=5, max_depth=4)
    iris_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    iris_data = pd.read_csv(iris_url, header=None)
    X = iris_data.drop(columns=[4])
    y = iris_data[4]

    # Map the labels to integers
    label_mapping = {label: idx for idx, label in enumerate(y.unique())}
    y = y.map(label_mapping)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit the model
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    accuracy = random_forest.score(y_test, y_pred)
    print(f"Accuracy on iris data: {accuracy:.2f}")

    # Test the predict_proba method
    y_pred_proba = random_forest.predict_proba(X_test)
    log_loss_value = random_forest.log_loss(y_test, y_pred_proba)
    print(f"Log loss on iris data: {log_loss_value:.4f}")