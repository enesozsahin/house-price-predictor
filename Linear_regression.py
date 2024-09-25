import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

class Linear_regression:

    def __init__(self, lr=0.0001, n_iters=10000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.rf_model = None  # To store the Random Forest model

    def fit(self, X, y):
        # Normalize data
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Gradient clipping to avoid large updates
            max_grad_value = 10
            dw = np.clip(dw, -max_grad_value, max_grad_value)
            db = np.clip(db, -max_grad_value, max_grad_value)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

        # Train Random Forest model and store it
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

    def predict_rf(self, X):
        if self.rf_model is None:
            raise ValueError("Random Forest model has not been trained. Call fit() first.")
        rf_y_pred = self.rf_model.predict(X)
        return rf_y_pred
