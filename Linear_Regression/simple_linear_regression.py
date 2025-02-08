import numpy as np

class LinearRegressionGD:
    def __init__(self):
        self.m = None  # Slope
        self.c = None  # Intercept
    
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        mean_X, mean_y = np.mean(X_train), np.mean(y_train)
        
        num = np.sum((X_train - mean_X) * (y_train - mean_y))
        den = np.sum((X_train - mean_X) ** 2)
        
        self.m = num / den
        self.c = mean_y - (self.m * mean_X)
        
        print(f"Slope (m): {np.round(self.m, 8)}, Intercept (c): {np.round(self.c, 8)}")
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        return self.m * X_test + self.c
    
    def r2_score(self, y_test, y_pred):
        ssr = np.sum((y_test - y_pred) ** 2)
        ssm = np.sum((y_test - np.mean(y_test)) ** 2)
        return 1 - (ssr / ssm)

# Example usage
slr = LinearRegressionGD()
slr.fit(X_train, y_train)

y_pred_test = slr.predict(X_test)
print("Predictions:", y_pred_test)
print("R2 Score:", slr.r2_score(y_test, y_pred_test))
