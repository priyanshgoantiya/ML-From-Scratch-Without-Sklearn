## softmax regression 
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Stability trick
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def compute_loss(self, Y, Y_pred):
        m = Y.shape[0]
        return -np.sum(Y * np.log(Y_pred + 1e-9)) / m  # Avoid log(0)
    
    def fit(self, X, Y):
        m, n = X.shape
        k = Y.shape[1]  # Number of classes
        self.weights = np.random.randn(n, k)
        self.bias = np.zeros((1, k))
        
        for epoch in range(self.epochs):
            logits = np.dot(X, self.weights) + self.bias
            Y_pred = self.softmax(logits)
            
            loss = self.compute_loss(Y, Y_pred)
            
            grad_w = np.dot(X.T, (Y_pred - Y)) / m
            grad_b = np.sum(Y_pred - Y, axis=0, keepdims=True) / m
            
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        Y_pred = self.softmax(logits)
        return np.argmax(Y_pred, axis=1)

# Generate synthetic data

Y = np.array(y).reshape(-1, 1)  # Ensure Y is a NumPy array and 2D
ohe = OneHotEncoder(sparse_output=False)  # Fix for newer sklearn versions
Y_one_hot = ohe.fit_transform(Y)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=0.2, random_state=42)

# Train softmax regression model
model = SoftmaxRegression(learning_rate=0.01, epochs=1000)
model.fit(X_train, Y_train)

# Predict and evaluate
Y_pred = model.predict(X_test)
Y_test_labels = np.argmax(Y_test, axis=1)
accuracy = np.mean(Y_pred == Y_test_labels)
print(f"Test Accuracy: {accuracy:.4f}")
