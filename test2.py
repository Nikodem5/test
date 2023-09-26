import numpy as np
from sklearn.linear_model import LinearRegression
import final as fi

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model1 = LinearRegression().fit(X, y)
model2 = fi.LinearRegression()
model2.fit(X, y, alpha=0.1, iters=1000)

print(model1.predict(np.array([[3, 5]])))
print(model2.predict(np.array([3, 5])))
