import pandas as pd

Variable1 = pd.read_csv("/Users/puspendra/Machine Learning and AI/HousingData.csv")
print(Variable1.info())
X = Variable1[["ZN", "INDUS", "RM", "AGE", "TAX"]]
Y = Variable1[["MEDV"]]

