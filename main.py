import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def unpickle(file):
    """Load a CIFAR-10 batch file using pickle."""
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding='bytes')


# ----------------------------------------------------------------------
# Load CIFAR-10 dataset
# ----------------------------------------------------------------------
data_dir = "cifar-10-python/cifar-10-batches-py"

# X = input data (images) | y = labels (correct answers, class 0-9)
X_train, y_train = [], []
for i in range(1, 6):
    batch = unpickle(os.path.join(data_dir, f"data_batch_{i}"))
    X_train.append(batch[b"data"])
    y_train.extend(batch[b"labels"])

X_train = np.concatenate(X_train)
y_train = np.array(y_train)

test_batch = unpickle(os.path.join(data_dir, "test_batch"))
X_test = test_batch[b'data']
y_test = np.array(test_batch[b'labels'])

# Use a subset for the assignment (full dataset would be too slow for KNN)
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

# Create smaller subset for testing
X_train_smaller = X_train[:5000]
y_train_smaller = y_train[:5000]
X_test_smaller = X_test[:1000]
y_test_smaller = y_test[:1000]

X_train_smaller = X_train_smaller.astype(np.float32)
X_test_smaller = X_test_smaller.astype(np.float32)


# ----------------------------------------------------------------------
# KNN classifier (from scratch)
# ----------------------------------------------------------------------
class KNN:
    def __init__(self, k=1, distance="L2"):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        # Store the training data (KNN is a "lazy learner" - no actual training)
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # Classify the labels of all test images
        predictions = []
        for i, test_image in enumerate(X_test):
            pred = self._predict_one(test_image)
            predictions.append(pred)

            if (i + 1) % 100 == 0:
                print(f" ...{i + 1}/{len(X_test)} Pictures processed")

        return np.array(predictions)

    def _predict_one(self, test_image):
        # Compute distance from this test image to all training images
        if self.distance == "L2":
            # Euclidean distance (L2 norm)
            distances = np.sqrt(np.sum((self.X_train - test_image) ** 2, axis=1))
        elif self.distance == "L1":
            # Manhattan distance (L1 norm)
            distances = np.sum(np.abs(self.X_train - test_image), axis=1)

        # Sort distances ascending and take indices of the k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]

        # Look up the labels of those k neighbors
        k_nearest_labels = self.y_train[k_nearest_indices]

        # Majority vote: pick the most frequent label
        prediction = np.bincount(k_nearest_labels).argmax()
        return prediction


# -----------------------------------------------------------------
# First quick test: K=3, L1
# ----------------------------------------------------------------------
print("start")
knn = KNN(3, distance="L1")
knn.fit(X_train, y_train)
preditions = knn.predict(X_test)

accuracy = np.mean(preditions == y_test)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


# ----------------------------------------------------------------------
# Compare different K values for L1 and L2 distance
# ----------------------------------------------------------------------
results_L1 = []
results_L2 = []

for k in range(1, 11, 2):

    # Use the smaller subset for testing
    X_train = X_train_smaller
    y_train = y_train_smaller
    X_test = X_test_smaller
    y_test = y_test_smaller

    # ----- L1 -----
    print(f"\nK={k}, computing L1 distance...")
    knn_l1 = KNN(k=k, distance="L1")
    knn_l1.fit(X_train, y_train)
    predictions_l1 = knn_l1.predict(X_test)
    acc_l1 = np.mean(predictions_l1 == y_test)
    results_L1.append(acc_l1)
    print(f"K={k}, L1: Accuracy = {acc_l1:.4f}")

    # ----- L2 -----
    print(f"\nK={k}, computing L2 distance...")
    knn_l2 = KNN(k=k, distance="L2")
    knn_l2.fit(X_train, y_train)
    predictions_l2 = knn_l2.predict(X_test)
    acc_l2 = np.mean(predictions_l2 == y_test)
    results_L2.append(acc_l2)
    print(f"K={k}, L2: Accuracy = {acc_l2:.4f}")

print("\n" + "="*40)
print("SUMMARY")
print("="*40)
print(f"{'K':<5}{'L1':<12}{'L2':<12}")
print("-"*40)
for idx, k in enumerate([1, 3, 5, 7, 9]):
    print(f"{k:<5}{results_L1[idx]:<12.4f}{results_L2[idx]:<12.4f}")


# ----------------------------------------------------------------------
# 5-Fold Cross-Validation
# --------------------------------------------------------------------
# Idea: Split the training data into 5 equal "folds".
# For each K value:
#   - Use 4 folds as training data, 1 fold as validation data.
#   - Rotate so every fold is used as validation exactly once.
#   - Average the 5 accuracies -> robust estimate for this K.
# This is more reliable than a single train/test split because the
# result does not depend on one specific lucky/unlucky split.
print("\n" + "="*40)
print("5-FOLD CROSS-VALIDATION (L2)")
print("="*40)

num_folds = 5
k_choices = [1, 3, 5, 7, 9]

# Split the training data into 5 folds along axis 0 (rows / samples)
X_train_folds = np.array_split(X_train_smaller, num_folds)
y_train_folds = np.array_split(y_train_smaller, num_folds)

# Dictionary: k -> list of 5 accuracies (one per fold)
k_to_accuracies = {}

for k in k_choices:
    print(f"\n--- Cross-validation for K={k} ---")
    accuracies = []

    for fold_idx in range(num_folds):
        # Use fold_idx as the validation fold, the rest as training
        X_val_fold = X_train_folds[fold_idx]
        y_val_fold = y_train_folds[fold_idx]

        # Combine the other 4 folds into one training set
        X_tr_folds = np.concatenate(
            [X_train_folds[i] for i in range(num_folds) if i != fold_idx]
        )
        y_tr_folds = np.concatenate(
            [y_train_folds[i] for i in range(num_folds) if i != fold_idx]
        )

        # Train KNN on the 4 folds and evaluate on the held-out fold
        knn_cv = KNN(k=k, distance="L2")
        knn_cv.fit(X_tr_folds, y_tr_folds)
        preds = knn_cv.predict(X_val_fold)
        acc = np.mean(preds == y_val_fold)
        accuracies.append(acc)
        print(f"  Fold {fold_idx + 1}/{num_folds}: Accuracy = {acc:.4f}")

    k_to_accuracies[k] = accuracies
    print(f"K={k}: mean = {np.mean(accuracies):.4f}, std = {np.std(accuracies):.4f}")

# Summary table of cross-validation
print("\n" + "="*40)
print("CROSS-VALIDATION SUMMARY")
print("="*40)
print(f"{'K':<5}{'Mean Acc':<12}{'Std':<12}")
print("-"*40)
for k in k_choices:
    mean_acc = np.mean(k_to_accuracies[k])
    std_acc = np.std(k_to_accuracies[k])
    print(f"{k:<5}{mean_acc:<12.4f}{std_acc:<12.4f}")

# Plot cross-validation results (scatter + mean line with error bars)
plt.figure(figsize=(8, 6))
for k in k_choices:
    accs = k_to_accuracies[k]
    plt.scatter([k] * len(accs), accs, color='blue', alpha=0.6)

mean_accuracies = [np.mean(k_to_accuracies[k]) for k in k_choices]
std_accuracies = [np.std(k_to_accuracies[k]) for k in k_choices]
plt.errorbar(k_choices, mean_accuracies, yerr=std_accuracies,
             fmt='-o', color='red', label='Mean ± Std')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('5-Fold Cross-Validation on CIFAR-10 (L2 distance)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cross_validation.png')
plt.show()

# Pick the best K based on cross-validation mean accuracy
best_k_cv = k_choices[int(np.argmax(mean_accuracies))]
print(f"\nBest K from cross-validation: {best_k_cv} "
      f"(mean accuracy = {max(mean_accuracies):.4f})")


# ----------------------------------------------------------------------
# Final evaluation on the held-out test set using the best K
# ----------------------------------------------------------------------
best_k_idx = np.argmax(results_L2)
best_k = [1, 3, 5, 7, 9][best_k_idx]
print(f"\nBest K (L2, single split): {best_k} with Accuracy {results_L2[best_k_idx]:.4f}")

# Use the K chosen by cross-validation for the final confusion matrix
knn_best = KNN(k=best_k_cv, distance='L2')
knn_best.fit(X_train_smaller, y_train_smaller)
best_predictions = knn_best.predict(X_test_smaller)

# Compute confusion matrix
cm = confusion_matrix(y_test_smaller, best_predictions)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (K={best_k_cv}, L2)')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()