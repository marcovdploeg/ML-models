# The Python file making a class for the gradient boosting classifier
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

class GradientBoostingClassifier:
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
        """Initialize the Gradient Boosting Classifier, 
        using the log loss function.
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
        # To hold the initial prediction for the binary case:
        self.y_pred_initial = None
        # To hold the trees after training for the binary case:
        self.tree_chain = []
        # To hold the model dictionary for the multi-class case:
        self.class_model_dict = {}
        self.n_classes = None  # to hold the number of classes in training
    
    def pseudo_residuals(
        self, 
        y_true: np.ndarray | pd.Series, 
        y_pred: np.ndarray | pd.Series
    ) -> np.ndarray | pd.Series:
        """Calculate the pseudo-residuals for the log loss
        that the decision tree will be fit to, which are
        the probability residuals in this case.

        Args:
            y_true (np.ndarray | pd.Series): True labels.
            y_pred (np.ndarray | pd.Series): Predicted probabilities.

        Returns:
            np.ndarray | pd.Series: Pseudo-residuals.
        """
        return (y_true - y_pred)

    def initial_prediction(
        self, 
        y_true: np.ndarray | pd.Series
    ) -> float:
        """Calculate the initial prediction for the gradient
        boosting algorithm, which is the log odds of the
        fraction of the class in the dataset for the log loss.

        Args:
            y_true (np.ndarray | pd.Series): True labels.

        Returns:
            float: Initial prediction.
        """
        initial_probability = np.mean(y_true)
        return np.log(initial_probability / (1 - initial_probability))
    
    def create_bootstrap_sample(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create a bootstrap sample of the given data.
        
        Args:
            X (pd.DataFrame): The original features dataset.
            y (pd.Series): The labels corresponding to the dataset.
        
        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - X_sample (pd.DataFrame): The bootstrap sample of
                    the dataset.
                - y_sample (pd.Series): The labels corresponding to
                    the bootstrap sample.
        """
        sample_size = int(self.subsample * len(X))

        # Generate a random sample without replacement
        sample_indices = np.random.choice(
            sample_size, size=sample_size, replace=False
        )
        X_sample, y_sample = X.iloc[sample_indices], y.iloc[sample_indices]
        return X_sample, y_sample
    
    def build_gradient_booster_binary(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None | tuple[float, list[DecisionTreeRegressor]]:
        """Build a gradient boosting model for binary classification by
        fitting a chain of decision trees to the pseudo-residuals.
        Note that this function is also used for multi-class classification,
        where it is called once for each class.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target labels, in the form of 0 and 1 or
                False and True.
        
        Returns:
            None: (in case of binary classification)
            tuple[float, list[DecisionTreeRegressor]]:
                (in case of multi-class classification)
                - y_pred_initial (float): The initial prediction for
                    this model.
                - tree_chain (list[DecisionTreeRegressor]): A list of
                    decision trees fitted to the pseudo-residuals.
        """
        # Initial prediction as log odds!
        self.y_pred_initial = self.initial_prediction(y)
        
        y_pred_for_residuals = self.y_pred_initial.copy()

        # In case we are fitting a multi-class case, the self.tree_chain 
        # needs to be reset for each class
        if self.n_classes > 2:
            self.tree_chain = []

        for _ in range(self.n_estimators):
            # Transform log odds to probability for the pseudo-residuals
            y_pred_for_residuals_prob = 1/(1 + np.exp(-y_pred_for_residuals))

            pseudo_residuals_i = self.pseudo_residuals(
                y, y_pred_for_residuals_prob
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

            # Add current predictions (to log odds) for the 
            # (transformation to probability) residuals in next loop
            y_pred_for_residuals += self.learning_rate * tree.predict(X)
        # In case we are fitting a multi-class case, the initial prediction
        # and tree chains should still be returned
        if self.n_classes > 2:
            return self.y_pred_initial, self.tree_chain
    
    def build_gradient_booster_multiclass(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        """Build a gradient boosting model for multi-class classification by
        fitting n_classes chains of decision trees to the pseudo-residuals.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target labels.
        """
        # Get the target labels with True/False for each class, 
        # through one-hot encoding
        y_one_hot = pd.get_dummies(y)
        for class_i in y_one_hot.columns:
            y_i = y_one_hot[class_i]

            # Treat each class as a binary case
            y_pred_initial_i, tree_chain_i = (
                self.build_gradient_booster_binary(X, y_i)
            )
            self.class_model_dict[class_i] = (
                y_pred_initial_i, tree_chain_i
            )
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        """Build a gradient boosting model for the right number of classes.
        For two classes, one chain of trees is fit to the pseudo-residuals. 
        For more than two classes, n_classes chains of trees are fit.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Target labels.
        """
        self.n_classes = y.nunique()
        if self.n_classes == 2:
            self.build_gradient_booster_binary(X, y)
        else:
            self.build_gradient_booster_multiclass(X, y)
    
    def predict_proba_binary(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make probability predictions using the fitted gradient boosting
        model for binary classification.

        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted probabilities for both classes,
                where the first column is for False/0,
                and the second column is for True/1.
        """
        y_pred = self.y_pred_initial.copy()
        for tree in self.tree_chain:
            y_pred += self.learning_rate * tree.predict(X)

        # Transform log odds output to probabilities
        y_pred_prob = 1 / (1 + np.exp(-y_pred))

        # Return as the probability for both classes
        return np.array([1 - y_pred_prob, y_pred_prob]).T
    
    def predict_proba_multiclass(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make probability predictions using the fitted gradient boosting
        model for multi-class classification.

        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted probabilities for each class, in the
                same order as in the training data (columns of y_one_hot).
        """
        predictions_log_odds = []
        for class_i in self.class_model_dict:
            y_pred_initial_i, tree_chain_i = self.class_model_dict[class_i]
            y_pred_i = y_pred_initial_i.copy()
            for tree in tree_chain_i:
                y_pred_i += self.learning_rate * tree.predict(X)
            
            predictions_log_odds.append(y_pred_i)
        
        # Use softmax to transform log odds to probabilities
        predictions_log_odds = np.array(predictions_log_odds)
        # Also subtract the max odds for numerical stability
        exp_predictions = np.exp(predictions_log_odds - 
                                 np.max(predictions_log_odds, axis=0))
        predictions_proba = exp_predictions / np.sum(exp_predictions, axis=0)
        return predictions_proba.T
    
    def predict_proba(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make probability predictions using the fitted gradient boosting
        model for the right number of classes.

        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted probabilities for each class.
        """
        if self.n_classes == 2:
            return self.predict_proba_binary(X)
        else:
            return self.predict_proba_multiclass(X)
    
    def predict_binary(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make class predictions using the fitted gradient boosting model
        for binary classification.

        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted class label.
        """
        y_pred = self.y_pred_initial.copy()
        for tree in self.tree_chain:
            y_pred += self.learning_rate * tree.predict(X)

        # Transform log odds output to probabilities
        y_pred_prob = 1 / (1 + np.exp(-y_pred))

        # Return the class 0 or 1 by rounding the probability
        return np.round(y_pred_prob).astype(int)
    
    def predict_multiclass(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make class predictions using the fitted gradient boosting model
        for multi-class classification.

        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted class labels.
        """
        predictions_proba = self.predict_proba_multiclass(X)
        # Get the class with the highest predicted probability for each sample
        predicted_classes = np.argmax(predictions_proba, axis=1)

        # Link these to the original labels (the columns of y_one_hot)
        class_labels = list(self.class_model_dict.keys())
        predicted_classes = [class_labels[i] for i in predicted_classes]
        return np.array(predicted_classes)
    
    def predict(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Make class predictions using the fitted gradient boosting model
        for the right number of classes.

        Args:
            X (pd.DataFrame): The data to make predictions on.

        Returns:
            np.ndarray: The predicted class labels.
        """
        if self.n_classes == 2:
            return self.predict_binary(X)
        else:
            return self.predict_multiclass(X)
        
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
        y_pred_proba: np.ndarray | pd.Series
    ) -> float:
        """Calculate the log loss between true labels and
        predicted probabilities.

        Args:
            y_true (np.ndarray | pd.Series): True labels, can be
                one-hot encoded.
            y_pred_proba (np.ndarray | pd.Series): Predicted probabilities
                for each class.

        Returns:
            float: Log loss value.
        """
        # Get epsilon from machine precision for y_pred_proba's dtype
        epsilon = np.finfo(y_pred_proba.dtype).eps

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
    print("Testing the GradientBoostingClassifier class.")
    
    gradient_booster = GradientBoostingClassifier(
        learning_rate=0.3,
        n_estimators=20,
        subsample=0.5
    )
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
    gradient_booster.fit(X_train, y_train)
    y_pred = gradient_booster.predict(X_test)
    accuracy = gradient_booster.score(y_test, y_pred)
    print(f"Accuracy on iris data: {accuracy:.2f}")

    # Test the predict_proba method
    y_pred_proba = gradient_booster.predict_proba(X_test)
    log_loss_value = gradient_booster.log_loss(y_test, y_pred_proba)
    print(f"Log loss on iris data: {log_loss_value:.4f}")