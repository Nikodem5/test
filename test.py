import FinalLib as ni
import numpy as np


X_train = np.array([
    [3, 1, 1180, 1],
    [3, 2.25, 2570, 2],
    [2, 1, 770, 1],
    [4, 3, 1960, 1],
    [3, 2, 1680, 1]
])

y_train = np.array([221900, 538000, 180000, 604000, 510000])

m, n = X_train.shape

model = ni.LinearRegression(np.random.rand(n), 100)
model.fit(X_train, y_train, alpha=0.0000001, iters=100)
print(model.predict(X_train[1]))
