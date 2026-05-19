# cifar10-knn-classifier

A K-Nearest Neighbors classifier implemented from scratch in NumPy, trained and evaluated on the CIFAR-10 image dataset.

## Features

- KNN implementation from scratch (no scikit-learn for the classifier itself)
- Supports both L1 (Manhattan) and L2 (Euclidean) distance metrics
- Compares accuracy across different K values (1, 3, 5, 7, 9)
- 5-fold cross-validation to select the best K
- Confusion matrix visualization
