# Machine-Learning-models

This repository contains the code for decision tree, random forest and gradient boosting 
classifier and regressor machine learning models. 
The classification decision tree uses the Gini impurity to improve the tree, 
while the regression tree uses the variance.
The classification models can predict both single class labels and probability distributions for the labels.
A maximum depth, minimum samples per split and minimum samples per leaf can be given to control the tree.
In the future, other evaluation metrics and options like other score criteria could be added.

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

Tests of both trees, forests and gradient boosters show that they generate predictions that are about as good as 
(and sometimes even slightly better than) Sklearn's implementations, but they are slower. 
Especially the ensemble methods that use many trees are rather slow compared to Sklearn.
