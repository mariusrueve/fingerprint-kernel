import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def tanimoto_kernel(X, Y):
    """
    Compute the Tanimoto similarity kernel between two sets of fingerprints.
    
    Parameters:
    - X: numpy array of shape (n_samples_X, n_features) representing the first set of fingerprints.
    - Y: numpy array of shape (n_samples_Y, n_features) representing the second set of fingerprints.
    
    Returns:
    - Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))
    
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dot_product = np.dot(X[i], Y[j])
            magnitude_a = np.sum(X[i])
            magnitude_b = np.sum(Y[j])
            tanimoto_similarity = dot_product / (magnitude_a + magnitude_b - dot_product)
            kernel_matrix[i, j] = tanimoto_similarity
    
    return kernel_matrix

# Example usage with synthetic data (as molecular fingerprint data is not readily available in sklearn)
# Create a synthetic binary classification dataset
X, y = make_classification(n_samples=100, n_features=10, n_informative=10, n_redundant=0, n_classes=2, random_state=42)

# Binarize the dataset to simulate molecular fingerprints
X = (X > np.mean(X, axis=0)).astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the custom SVM with the Tanimoto kernel
class TanimotoSVC(SVC):
    def __init__(self, **kwargs):
        super().__init__(kernel='precomputed', **kwargs)
    
    def fit(self, X, y):
        self.X_fit_ = X
        return super().fit(tanimoto_kernel(X, X), y)
    
    def predict(self, X):
        return super().predict(tanimoto_kernel(X, self.X_fit_))

# Initialize and train the Tanimoto SVM
tanimoto_svm = TanimotoSVC(C=1.0)
tanimoto_svm.fit(X_train, y_train)

# Predict on the test set
y_pred = tanimoto_svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
