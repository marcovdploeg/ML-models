# The Python file making a class for the decision tree regressor model.
# Here we add the random selection of features at each node, as 
# required for the Random Forest model.

import numpy as np
import pandas as pd

class DecisionTreeRegressorModified:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 verbose=False, threshold=0.0, num_features=None, random_state=42):
        """
        Initialize the Decision Tree Classifier.

        Parameters:
        max_depth (int, optional): Maximum depth of the tree.
                                   Default is None (no limit).
        min_samples_split (int, optional): Minimum number of samples required
                                     to split an internal node. Default is 2.
        min_samples_leaf (int, optional): Minimum number of samples required
                                         to be at a leaf node. Default is 1.
        verbose (bool, optional): If True, print the reasons for stopping.
                                  Default is False.
        threshold (float, optional): The threshold to compare against
                                     for increasing score. Default=0.0.
        num_features (int, optional): Number of features to consider in each
                                      node. If None, one-third of the 
                                      total number of features rounded down
                                      is used. Default is None.
        random_state (int, optional): Random seed for reproducibility.
                                      Default is 42.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose = verbose
        self.threshold = threshold
        self.num_features = num_features
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.tree = None  # to hold the tree structure after fitting

    def var_weighted(self, y_left, y_right):
        """
        Calculate the weighted variance after a split.
        
        Parameters:
        y_left (array-like): Array of labels for the left dataset.
        y_right (array-like): Array of labels for the right dataset.
        
        Returns:
        float: Weighted variance value.
        """
        N_left = len(y_left)
        N_right = len(y_right)
        N_total = N_left + N_right

        var_left = np.var(y_left) if N_left > 1 else 0.0
        var_right = np.var(y_right) if N_right > 1 else 0.0
        
        var_weighted = (N_left / N_total) * var_left + \
                       (N_right / N_total) * var_right
        return var_weighted
    
    def determine_continuous_split(self, X_feature):
        """
        Determine the split points for a continuous feature, being the
        first quantile, median, and third quantile. If there are 3 or less
        unique values, this is not useful, so return the median instead.

        Parameters:
        X_feature (Series): Series of feature values.

        Returns:
        (list or float): List of split points [first quantile, median, third quantile],
                         median if not enough data to determine quantiles.
        """
        if len(X_feature.unique()) <= 3:
            median = np.median(X_feature)
            return median  # Not enough data to determine quantiles
        else:
            first_quantile = np.quantile(X_feature, 0.25)
            median = np.median(X_feature)
            third_quantile = np.quantile(X_feature, 0.75)
            return [first_quantile, median, third_quantile]
        
    def split_node_and_find_best_split(self, X_feature, y):
        """
        Split a node based on the feature values in X_feature while
        determining the best split point based on variance for
        continuous features.
        
        Parameters:
        X_feature (Series): Series of feature values.
        y (Series): Series of target values.
        
        Returns:
        Series: Two Series representing the left and right splits for y.
        (float or None): The best split value for continuous features, 
                         None for boolean features.
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

            # If split points are a single value (median), use that
            if isinstance(split_points, float):
                median = split_points
                y_left = y[X_feature.index[X_feature <= median]]
                y_right = y[X_feature.index[X_feature > median]]
                best_split_value = median  # Need to return this
            else:
                # Figure out which split is best, using the lowest variance
                best_var = float('inf')
                for split_value in split_points:
                    y_left = y[X_feature.index[X_feature <= split_value]]
                    y_right = y[X_feature.index[X_feature > split_value]]
                    var = self.var_weighted(y_left, y_right)
                    if var < best_var:
                        best_var = var
                        best_split_value = split_value
                # Now split based on the best split value found
                y_left = y[X_feature.index[X_feature <= best_split_value]]
                y_right = y[X_feature.index[X_feature > best_split_value]]
        
        return y_left, y_right, best_split_value
    
    def determine_best_split(self, X, y):
        """
        Determine the best split for a dataset based on variance.
        
        Parameters:
        X (DataFrame): Feature dataframe.
        y (Series): Target values.
        
        Returns:
        tuple: The best feature and the best split value 
               (None for boolean features).
        """
        best_var = float('inf')

        for column in X.columns:
            X_feature = X[column]
            y_left, y_right, best_split_value = \
                self.split_node_and_find_best_split(X_feature, y)
            
            # Calculate variance for the split
            var = self.var_weighted(y_left, y_right)
            if var < best_var:
                best_var = var
                best_split_feature = (column, best_split_value)
        return best_split_feature
    
    def check_var_decrease_threshold(self, y, y_left, y_right):
        """
        Calculate the variance decrease from a split and if it meets
        a threshold.
        
        Parameters:
        y (array-like): Array of target values before the split.
        y_left (array-like): Array of target values for the left dataset.
        y_right (array-like): Array of target values for the right dataset.
        threshold (float): The threshold to compare against.
        
        Returns:
        bool: True if the variance decrease is greater than
              the threshold, False otherwise.
        """
        var_before = np.var(y)
        var_after = self.var_weighted(y_left, y_right)
        
        return (var_before - var_after) > self.threshold
    
    def split_node_given_best_split(self, X, y, feature, best_split_value):
        """
        Split a node based on the feature values in X_feature given
        the predetermined best split point.
        
        Parameters:
        X (DataFrame): Array of feature values.
        y (Series): Array of target values.
        best_split_value (float or None): The best split value for
                   continuous features, None for boolean features.
        
        Returns:
        DataFrame: Two DataFrames representing the left and right splits for X.
        Series: Two Series representing the left and right splits for y.
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
    
    def select_random_features(self, X):
        """
        Select a random subset of features from the DataFrame X.
        
        Parameters:
        X (DataFrame): Feature dataframe.
        
        Returns:
        DataFrame: A DataFrame containing the selected features.
        """
        total_features = X.shape[1]
        if self.num_features is None:
            num_features = int(total_features / 3)
            # Unlike with a square root, this number can be smaller than 1,
            # so that it would be rounded down to 0, which is not useful.
            # So we ensure it is at least 1
            num_features = max(1, num_features)
        elif self.num_features > total_features:
            num_features = total_features  # ensure we don't exceed available features
        else:
            num_features = self.num_features
        
        # Randomly select features to consider for the split, without replacement
        selected_features = np.random.choice(X.columns, num_features, replace=False)
        X_selected = X[selected_features]
        return X_selected
    
    def build_tree(self, X, y, depth=0):
        """
        Build up the tree recursively by fitting the training data.
        This needs to start with depth at 0, which is passed
        as an argument to make the recursion work.

        Parameters:
        X (DataFrame): Feature dataframe.
        y (Series): Target values.

        Returns:
        dict: A dictionary representing the decision tree.
        """
        # First check all stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth):
            if self.verbose:
                print(f"Stopping at depth {depth} due to max_depth limit.")
            mean_value = y.mean()
            return {'value': mean_value, 'is_leaf': True}
        elif len(y) < self.min_samples_split:
            if self.verbose:
                print(f"Stopping at depth {depth} due to min_samples_split limit.")
            mean_value = y.mean()
            return {'value': mean_value, 'is_leaf': True}

        # For the variance score increase and min_samples_leaf criteria,
        # we need to do the split first
        # And now that we select random features at each node,
        # we need to select a random subset of features here too
        # Check it every time in case we dropped a boolean feature
        X_selected = self.select_random_features(X)

        # Then find the best split for the selected features only
        column, split_value = self.determine_best_split(X_selected, y)
        # Because we do need the full feature set for the next recursion,
        # and X_left, X_right are not used otherwise and y_left, y_right
        # are the same for both, we do split on the full X here
        X_left, X_right, y_left, y_right = \
            self.split_node_given_best_split(X, y, column, split_value)
        if len(y_left) < self.min_samples_leaf or \
           len(y_right) < self.min_samples_leaf:
            if self.verbose:
                print(f"Stopping at depth {depth} due to min_samples_leaf limit.")
            mean_value = y.mean()
            return {'value': mean_value, 'is_leaf': True}
        
        better_score = self.check_var_decrease_threshold(y, y_left, y_right)
        if not better_score:
            if self.verbose:
                print(f"Stopping at depth {depth} as no better score is found.")
            mean_value = y.mean()
            return {'value': mean_value, 'is_leaf': True}
        
        # If none of the criteria are met, we continue here
        # If it is a boolean feature we can just take it out,
        # as splitting on it again is nonsense
        if split_value is None:
            X_left = X_left.drop(columns=[column])
            X_right = X_right.drop(columns=[column])

        # Recursively build the left and right subtrees, adding 1 to the depth
        left_subtree = self.build_tree(X_left, y_left, depth=depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth=depth + 1)

        # Return the current node with its left and right subtrees;
        # in the end this contains the full tree
        return {
            'feature': column,
            'split': split_value,
            'is_leaf': False,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """
        Fit the decision tree regressor to the training data.
        
        Parameters:
        X (DataFrame): Feature dataframe.
        y (Series): Target values.
        """
        self.tree = self.build_tree(X, y)
    
    def predict(self, X):
        """
        Predict the value for each sample in X using the decision tree.
        
        Parameters:
        X (DataFrame): Feature dataframe for which to make predictions.
        
        Returns:
        array-like: Predicted value for each sample in X.
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
                # After going down, if the node is still not a leaf, repeat this
            # When we reach a leaf node, append the predicted value
            predictions.append(node['value'])
        
        return np.array(predictions)
    
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
    print("Testing the DecisionTreeRegressorModified class.")

    decision_tree = DecisionTreeRegressorModified(max_depth=3, num_features=None)

    iris_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
    iris_data = pd.read_csv(iris_url, header=None)
    # Take the first column as target variable for this regression task
    X = iris_data.drop(columns=[0])
    y = iris_data[0]

    # Map the labels to integers
    label_mapping = {label: idx for idx, label in enumerate(X[4].unique())}
    X[4] = X[4].map(label_mapping)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the model
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    rmse = decision_tree.score(y_test, y_pred)
    print(f"Root Mean Square Error on iris data: {rmse:.4f}")
    # Note that we indeed see a change in error now due to
    # the usage of less and random features at each node