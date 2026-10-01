# The Python file making a class for the decision tree classifier
# model thought up in the notebook.

import numpy as np
import pandas as pd

class DecisionTreeClassifier:
    def __init__(
        self, 
        max_depth: int | None = 3, 
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1, 
        verbose: bool = False, 
        threshold: float = 0.0
    ) -> None:
        """Initialize the Decision Tree Classifier.

        Args:
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
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.threshold = threshold
        self.tree = None  # to hold the tree structure after fitting
        # To hold the number of labels in the training data:
        self.N_original_labels = None
    
    def gini_impurity(
        self, 
        y: pd.Series
    ) -> float:
        """Calculate the Gini impurity for a set of labels.
        
        Args:
            y (pd.Series): Array of labels.
        
        Returns:
            float: Gini impurity value.
        """
        if len(y) == 0:
            return 0.0
        
        # Count occurrences of each label
        counts = y.value_counts(normalize=True)
        gini = 1 - np.sum(counts ** 2)
        return gini
    
    def gini_weighted(
        self, 
        y_left: pd.Series, 
        y_right: pd.Series
    ) -> float:
        """Calculate the weighted Gini impurity after a split.
        
        Args:
            y_left (pd.Series): Array of labels for the left dataset.
            y_right (pd.Series): Array of labels for the right dataset.
        
        Returns:
            float: Weighted Gini impurity value.
        """
        N_left = len(y_left)
        N_right = len(y_right)
        N_total = N_left + N_right

        gini_left = self.gini_impurity(y_left)
        gini_right = self.gini_impurity(y_right)
        
        gini_weighted = (
            (N_left / N_total) * gini_left + (N_right / N_total) * gini_right
        )
        return gini_weighted
    
    def determine_continuous_split(
        self, 
        X_feature: pd.Series
    ) -> list | float:
        """Determine the split points for a continuous feature, being the
        first quantile, median, and third quantile. If there are 3 or less
        unique values, this is not useful, so return the median instead.

        Args:
            X_feature (pd.Series): Series of feature values.

        Returns:
            list | float: List of split points
                [first quantile, median, third quantile],
                or median if not enough data to determine quantiles.
        """
        if len(X_feature.unique()) <= 3:
            median = np.median(X_feature)
            return median  # Not enough data to determine quantiles
        else:
            first_quantile = np.quantile(X_feature, 0.25)
            median = np.median(X_feature)
            third_quantile = np.quantile(X_feature, 0.75)
            return [first_quantile, median, third_quantile]

    def split_node_and_find_best_split(
        self, 
        X_feature: pd.Series, 
        y: pd.Series
    ) -> tuple[pd.Series, pd.Series, float | None]:
        """Split a node based on the feature values in X_feature while
        determining the best split point based on Gini impurity for
        continuous features.
        
        Args:
            X_feature (pd.Series): Series of feature values.
            y (pd.Series): Labels.
        
        Returns:
            tuple[pd.Series, pd.Series, float | None]: A tuple containing:
                - y_left (pd.Series): Labels of the left split for y.
                - y_right (pd.Series): Labels of the right split for y.
                - best_split_value (float | None): The best split value for
                    continuous features, None for boolean features.
        """
        # First figure out if the feature is continuous or boolean
        # This works for both 0/1 and True/False as values
        if set(X_feature.unique()).issubset({0, 1}):
            # Boolean feature
            y_left = y[X_feature.index[X_feature == 0]]
            y_right = y[X_feature.index[X_feature == 1]]
            best_split_value = None  # No split value for boolean features
        else:
            # Continuous feature
            split_points = self.determine_continuous_split(X_feature)

            # If split points is a single value (median), use that
            if isinstance(split_points, float):
                median = split_points
                y_left = y[X_feature.index[X_feature <= median]]
                y_right = y[X_feature.index[X_feature > median]]
                best_split_value = median  # Need to return this
            else:
                # Figure out which split is best
                best_gini = float('inf')
                for split_value in split_points:
                    y_left = y[X_feature.index[X_feature <= split_value]]
                    y_right = y[X_feature.index[X_feature > split_value]]
                    gini = self.gini_weighted(y_left, y_right)
                    if gini < best_gini:
                        best_gini = gini
                        best_split_value = split_value
                # Now split based on the best split value found
                y_left = y[X_feature.index[X_feature <= best_split_value]]
                y_right = y[X_feature.index[X_feature > best_split_value]]
        
        return y_left, y_right, best_split_value

    def determine_best_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> tuple[str, float | None]:
        """Determine the best split for a dataset based on Gini impurity.
        
        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Labels.
        
        Returns:
            tuple[str, float | None]: A tuple containing:
                - column (str): The best feature to split on.
                - best_split_value (float | None): The best split value for
                    continuous features, None for boolean features.
        """
        best_gini = float('inf')

        for column in X.columns:
            X_feature = X[column]
            y_left, y_right, best_split_value = (
                self.split_node_and_find_best_split(X_feature, y)
            )
            
            # Calculate Gini impurity for the split
            gini = self.gini_weighted(y_left, y_right)
            if gini < best_gini:
                best_gini = gini
                best_split_feature = (column, best_split_value)
        return best_split_feature
    
    def check_impurity_decrease_threshold(
        self, 
        y: pd.Series, 
        y_left: pd.Series, 
        y_right: pd.Series
    ) -> bool:
        """Calculate the impurity decrease from a split and if it meets
        a threshold.
        
        Args:
            y (pd.Series): Array of labels before the split.
            y_left (pd.Series): Array of labels for the left dataset.
            y_right (pd.Series): Array of labels for the right dataset.
        
        Returns:
            bool: True if the impurity decrease is greater than 
                the threshold, False otherwise.
        """
        gini_before = self.gini_impurity(y)
        gini_after = self.gini_weighted(y_left, y_right)
        return (gini_before - gini_after) > self.threshold
    
    def split_node_given_best_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        feature: str, 
        best_split_value: float | None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split a node based on the feature values in X[feature] given 
        the predetermined best split point.
        
        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Labels.
            feature (str): The feature to split on.
            best_split_value (float | None): The best split value for
                continuous features, None for boolean features.
        
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                - X_left (pd.DataFrame): Feature values for the left split.
                - X_right (pd.DataFrame): Feature values for the right split.
                - y_left (pd.Series): Labels for the left split.
                - y_right (pd.Series): Labels for the right split.
        """
        X_feature = X[feature]
        if best_split_value is None:
            # Boolean feature
            X_left = X[X_feature == 0]
            X_right = X[X_feature == 1]
            y_left = y[X.index[X_feature == 0]]
            y_right = y[X.index[X_feature == 1]]
        else:
            # Continuous feature
            X_left = X[X_feature <= best_split_value]
            X_right = X[X_feature > best_split_value]
            y_left = y[X.index[X_feature <= best_split_value]]
            y_right = y[X.index[X_feature > best_split_value]]
        
        return X_left, X_right, y_left, y_right

    def build_tree(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        depth: int = 0
    ) -> dict:
        """Build up the tree recursively by fitting the training data.
        This needs to start with depth at 0, which is passed as
        an argument to make the recursion work.

        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Labels.

        Returns:
            dict: A dictionary representing the decision tree.
        """
        # First check all stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth):
            if self.verbose:
                print(f"Stop at depth {depth} due to max_depth limit.")
            proba_dist = y.value_counts(normalize=True).to_dict()
            return {'probas': proba_dist, 'is_leaf': True}
        elif len(y) < self.min_samples_split:
            if self.verbose:
                print(f"Stopping at depth {depth} due to min_samples_split limit.")
            proba_dist = y.value_counts(normalize=True).to_dict()
            return {'probas': proba_dist, 'is_leaf': True}

        # For the Gini score increase and min_samples_leaf criteria, 
        # we need to do the split first
        column, split_value = self.determine_best_split(X, y)
        X_left, X_right, y_left, y_right = (
            self.split_node_given_best_split(X, y, column, split_value)
        )
        if (
            len(y_left) < self.min_samples_leaf
            or len(y_right) < self.min_samples_leaf
        ):
            if self.verbose:
                print(f"Stopping at depth {depth} due to min_samples_leaf limit.")
            proba_dist = y.value_counts(normalize=True).to_dict()
            return {'probas': proba_dist, 'is_leaf': True}
        
        better_score = (
            self.check_impurity_decrease_threshold(y, y_left, y_right)
        )
        if not better_score:
            if self.verbose:
                print(f"Stop at depth {depth} as no better score is found.")
            proba_dist = y.value_counts(normalize=True).to_dict()
            return {'probas': proba_dist, 'is_leaf': True}
        
        # If none of the criteria are met, we continue here
        # If it is a boolean feature we can just take it out, 
        # as splitting on it again is nonsense
        if split_value is None:
            X_left = X_left.drop(columns=[column])
            X_right = X_right.drop(columns=[column])

        # Recursively build the left and right subtrees, adding 1 to the depth
        left_subtree = self.build_tree(X_left, y_left, depth=depth+1)
        right_subtree = self.build_tree(X_right, y_right, depth=depth+1)

        # Return the current node with its left and right subtrees; 
        # in the end this contains the full tree
        return {
            'feature': column,
            'split': split_value,
            'is_leaf': False,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> None:
        """Fit the decision tree classifier to the training data.
        
        Args:
            X (pd.DataFrame): Feature dataframe.
            y (pd.Series): Labels.
        """
        self.tree = self.build_tree(X, y)
        self.N_original_labels = len(y.unique())
    
    def predict(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Predict the class for each sample in X using the decision tree.
        
        Args:
            X (pd.DataFrame): Feature dataframe for which to make predictions.
        
        Returns:
            np.ndarray: Predicted classes for each sample in X.
        """
        predictions = []
        for _, row in X.iterrows():
            node = self.tree  # start at the root node
            while not node['is_leaf']:
                feature = node['feature']
                split_value = node['split']
                data_feature_value = row[feature]
                if split_value is None:  # Boolean feature
                    if data_feature_value == 0:
                        # We trained the model to go left for 0
                        node = node['left']
                    else:
                        node = node['right']
                else:  # Continuous feature
                    if data_feature_value <= split_value:
                        # We trained the model to go left for <= split_value
                        node = node['left']
                    else:
                        node = node['right']
                # After going down, if the node is not a leaf, repeat this
            # At a leaf node, get the class with the highest probability
            # and append the predicted class to the predictions
            proba_dist = node['probas']
            predicted_class = max(proba_dist, key=proba_dist.get)
            predictions.append(predicted_class)
        return np.array(predictions)
    
    def predict_proba(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """Predict the probability distribution for each sample in X
        using the decision tree.
        
        Args:
            X (pd.DataFrame): Feature dataframe for which to make predictions.
        
        Returns:
            np.ndarray: Predicted probability distributions for each sample.
        """
        predictions = []
        for _, row in X.iterrows():
            node = self.tree  # start at the root node
            while not node['is_leaf']:
                feature = node['feature']
                split_value = node['split']
                data_feature_value = row[feature]
                if split_value is None:  # Boolean feature
                    if data_feature_value == 0:
                        # We trained the model to go left for 0
                        node = node['left']
                    else:
                        node = node['right']
                else:  # Continuous feature
                    if data_feature_value <= split_value:
                        # We trained the model to go left for <= split_value
                        node = node['left']
                    else:
                        node = node['right']
                # After going down, if the node is not a leaf, repeat this
            # At a leaf node, take the predicted classes probabilities
            proba_dist = node['probas']
            # Convert the probabilities to a list in
            # the order of the original labels
            all_probas = [
                proba_dist.get(i, 0) for i in range(self.N_original_labels)
            ]
            predictions.append(all_probas)
        return np.array(predictions)

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
    print("Testing the DecisionTreeClassifier class.")

    decision_tree = DecisionTreeClassifier(max_depth=4)
    iris_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    iris_data = pd.read_csv(iris_url, header=None)
    X = iris_data.drop(columns=[4])
    y = iris_data[4]

    # Map the labels to integers
    label_mapping = {label: idx for idx, label in enumerate(y.unique())}
    y = y.map(label_mapping)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit the model
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    accuracy = decision_tree.score(y_test, y_pred)
    print(f"Accuracy on iris data: {accuracy:.2f}")

    # Test the predict_proba method
    y_pred_proba = decision_tree.predict_proba(X_test)
    log_loss_value = decision_tree.log_loss(y_test, y_pred_proba)
    print(f"Log loss on iris data: {log_loss_value:.4f}")