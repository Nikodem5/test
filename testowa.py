import numpy as np

class LinearRegression:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def predict(self, x):
        return np.dot(self.w, x) + self.b

    def compute_cost(self, X, y):
        m, n = X.shape
        err = 0
        for i in range(m):
            err += np.square(self.predict(X[i]) - y[i])
        err = err / (2*m)

        return err

    def compute_gradient(self, X, y):
        m, n = X.shape
        dj_db = 0
        dj_dw = np.zeros(n)

        for i in range(m):
            dj_db += self.predict(X[i]) - y[i]
            for j in range(n):
                dj_dw[j] += (self.predict(X[i]) - y[i]) * X[i][j]
        dj_db = dj_db / m
        dj_dw = dj_dw / m

        return dj_dw, dj_db

    def gradient_descent(self, X, y, alpha, iters):
        for i in range(iters):
            dj_dw, dj_db = self.compute_gradient(X, y)

            w_tmp = self.w - (alpha * dj_dw)
            b_tmp = self.b - (alpha * dj_db)
            self.w = w_tmp
            self.b = b_tmp

            if i % 10 == 0:
                print(f'iteration: {i} cost: {self.compute_cost(X, y)}')

        return self.w, self.b

    def fit(self, X, y, alpha=0.01, iters=10):
        m, n = X.shape
        self.w = np.random.rand(n)
        self.b = 100
        self.w, self.b = self.gradient_descent(X, y, alpha, iters)
