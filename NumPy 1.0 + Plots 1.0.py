import numpy as np
import matplotlib.pyplot as plots

Variable = np.random.randint(1,10,10)
print(Variable)
Variable2 = [1,3,4,7,9,13,14,16,19,20]
Array1 = np.array(Variable2)
Average1 = Array1.mean()
print(Variable2)
Variable3 = [21,25,29,30,33,34,35,38,40,44]
Array2 = np.array(Variable3)
Average2 = Array2.mean()
print(Variable3)
Variable4 = plots.scatter(Variable2, Variable3, color = "Gold")
M = (np.sum((Array1 - Average1)*(Array2 - Average2))/np.sum((Array1 - Average1)**2))
print(M)
C = (Average2 - M*Average1)
print(C)
Y = M*Array1 + C
X = np.sqrt(np.mean((Y - Array2)**2))
print(X)
Y = plots.plot(Array1, Y, color = "Gold")
Y = plots.show()

from sklearn.linear_model import LinearRegression
Sketch = LinearRegression()
VariableX = Array1.reshape(-1,1)
