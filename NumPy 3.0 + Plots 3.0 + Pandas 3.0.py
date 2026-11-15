import numpy as np
import matplotlib.pyplot as plots
import pandas as pd

Variable = pd.read_csv("/Users/puspendra/Machine Learning and AI/student_performance_dirty.csv")
Variable1 = Variable["Marks"]
Array1 = np.array(Variable1)
Average1 = Array1.mean()
Variable2 = Variable["StudyHours"]
Array2 = np.array(Variable2)
Average2 = Array2.mean()
Variable3 = plots.scatter(Variable2, Variable1, color = "Blue")

from sklearn.linear_model import LinearRegression
Sketch = LinearRegression()
VariableX = Array1.reshape(-1,1)
Sketch.fit(VariableX, Array2)
print(Sketch.coef_)
print(Sketch.intercept_)
VariableY = Sketch.predict(VariableX)

Variable3 = plots.plot(VariableY, VariableX, color = "Blue")
VariableZ = plots.show()
