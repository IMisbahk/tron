# tron/models/regression/linearRegression.py

import numpy as np
import uuid
import time

class LinearRegression:
    

    def __init__(
        self,
        learningRate=0.01,
        epochs=1000,
        tolerance=1e-6,
        fitIntercept=True,
        regularization=None,     
        regStrength=0.01,
        origin="architect",
        owner="system",
        reason="init",
        label=None
    ):
        self.learningRate = learningRate
        self.epochs = epochs
        self.tolerance = tolerance
        self.fitIntercept = fitIntercept
        self.regularization = regularization
        self.regStrength = regStrength

        self.origin = origin
        self.owner = owner
        self.reason = reason
        self.label = label or f"linreg-{str(uuid.uuid4())[:8]}"
        self.created = time.time()

        self.weights = None
        self.bias = 0.0
        self.lossHistory = []
        self.traceLog = []
        self.metadata = {
            "label": self.label,
            "origin": self.origin,
            "owner": self.owner,
            "reason": self.reason
        }

    def computeLoss(self, y, yHat):
        m = len(y)
        error = yHat - y
        mse = np.mean(error ** 2)
        if self.regularization == 'l2':
            mse += (self.regStrength / (2 * m)) * np.sum(np.square(self.weights))
        elif self.regularization == 'l1':
            mse += (self.regStrength / m) * np.sum(np.abs(self.weights))
        return mse

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape

        if self.fitIntercept:
            X = np.hstack([np.ones((m, 1)), X])

        self.weights = np.zeros(X.shape[1])

        for epoch in range(self.epochs):
            yHat = np.dot(X, self.weights)
            error = yHat - y
            gradient = np.dot(X.T, error) / m

            if self.regularization == 'l2':
                gradient += (self.regStrength / m) * self.weights
            elif self.regularization == 'l1':
                gradient += (self.regStrength / m) * np.sign(self.weights)

            self.weights -= self.learningRate * gradient
            loss = self.computeLoss(y, yHat)

            self.lossHistory.append(loss)
            self.traceLog.append({
                "epoch": epoch,
                "loss": loss,
                "weights": self.weights.copy(),
                "timestamp": time.time()
            })

            if epoch > 0 and abs(self.lossHistory[-2] - loss) < self.tolerance:
                break

    def predict(self, X):
        X = np.array(X, dtype=float)
        if self.fitIntercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.weights)

    def score(self, X, y):
        y = np.array(y)
        yHat = self.predict(X)
        error = yHat - y
        return 1 - (np.sum(error**2) / np.sum((y - np.mean(y))**2))  # R^2 Score

    def observe(self, includeTrace=False):
        obs = {
            "label": self.label,
            "origin": self.origin,
            "owner": self.owner,
            "created": self.created,
            "learningRate": self.learningRate,
            "epochs": self.epochs,
            "tolerance": self.tolerance,
            "regularization": self.regularization,
            "regStrength": self.regStrength,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "lossHistory": self.lossHistory[-10:]
        }
        if includeTrace:
            obs["traceLog"] = self.traceLog
        return obs

    def __repr__(self):
        return f"<TRON LinearRegression | {self.label} | Origin: {self.origin}>"
