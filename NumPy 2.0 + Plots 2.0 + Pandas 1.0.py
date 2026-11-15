import numpy as np
import matplotlib.pyplot as plots
import pandas as pd

Variable1 = pd.read_csv("/Users/puspendra/Machine Learning and AI/study_hours_marks.csv")

Variable2 = Variable1["X"]
Array1 = np.array(Variable2)
Average1 = Array1.mean()
print(Variable1)
Variable3 = Variable1["Y"]
Array2 = np.array(Variable3)
Average2 = Array2.mean()
print(Variable3)
Variable4 = plots.scatter(Variable2, Variable3, color = "gold")
M = (np.sum((Array1 - Average1)*(Array2 - Average2))/np.sum((Array1 - Average1)**2))
print(M)
C = (Average2 - M*Average1)
print(C)
Y = M*Array1 + C
print(Y)
X = plots.plot(Array1, Y, color = "Gold")
X = plots.show()
