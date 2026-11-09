class NormalEquation:

    def __init__(self):
        self.X = None
        self.y = None
        self.theta = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.X = X
        self.y = y
        m = X.shape[0] # Number of samples
        X_b = np.c_[np.ones((m, 1)), X] # Add bias column
        X_transpose = X_b.T # Normal Equation
        self.theta = (
            np.linalg.pinv(X_transpose @ X_b)
            @ X_transpose
            @ y
        )

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.theta is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta