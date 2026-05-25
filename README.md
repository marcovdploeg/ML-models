# Machine Learning models

This repository contains the code for decision tree, random forest and gradient boosting 
classifier and regressor machine learning models. 

## Contents

The decision\_tree\_classifier and decision\_tree\_regressor directories contain a Jupyter notebook explaining 
how each tree works and a Python file with a class that contains the tree algorithm and 
could be imported into other scripts.
The random\_forest\_classifier and random\_forest\_regressor directories similarly contain a Jupyter notebook 
explaining how the random forest works and a Python file with a class that contains the forest algorithm and 
could be imported into other scripts. 
Additionally, there are modified decision tree algorithms which are needed for the random forest implementation.
The gradient\_boosting\_regressor and gradient\_boosting\_classifier directories also contain an explanatory 
Jupyter notebook and importable Python script, while importing the original decision tree regressor to use 
as their weak learner.

## Decision tree model details

The classification decision tree uses the Gini impurity as the metric to improve the tree with, 
while the regression tree uses the variance.
The classification models can predict both single class labels and probability distributions for the labels.
A maximum depth, minimum samples per split, minimum samples per leaf and minimum increase threshold 
can be given to control the tree.
In the future, other evaluation metrics and options like other score criteria could be added.

## Random forest model details

The random forest models are built using the corresponding implementation of the decision tree.
Here we do use a modified version of the trees, to add more randomness to each tree node during training.
Each tree is trained on different bootstrap samples of the data.
For the classifier, the random forest prediction is determined by a majority vote from the decision trees 
for class labels, and by an average of decision tree probabilities for the probability distribution.
For the regressor, the random forest prediction is simply the average of the tree predictions.
On top of the decision tree parameters, the number of trees to use now also needs to be given, and 
optionally the number of features to use in each tree node.

## Gradient boosting model details

The gradient boosting models are built using the implementation of the decision tree regressor.
The regression model uses the mean squared error loss function, while the classification model uses 
the logarithmic loss function.
The decision trees are trained successively on the resulting pseudo-residuals to generate predictions.
On top of the decision tree parameters, here you also need to give the number of estimators (i.e. trees) 
to use, the learning rate, and the fraction of the data to be used for training each estimator.
In the future, other loss functions and base models for the estimator could be added.

## Test results

Tests of both trees, forests and gradient boosters show that they generate predictions that are about as good as 
(and sometimes even slightly better than) Sklearn's implementations, but they are slower. 
Especially the ensemble methods that use many trees are rather slow compared to Sklearn.
