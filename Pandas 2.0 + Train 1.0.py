import pandas as pd


#CRIM = Crime rate
#ZN = Proportion of residential landson
#INDUS = Proportion of non retail business acres
#CHAS = Whether sharing boundary with Charles River
#NOX = Nitric Oxide concentration
#RM = Number of rooms
#AGE = Age of the house
#DIS = Distance to enployement centers
#RAD = Accessibility to Highways
#TAX = Tax rate
#PTRATIO = Pupil teacher ratio
#LSTAT = Percentage or lower status population
#MEDV = Median house value
Variable1 = pd.read_csv("/Users/puspendra/Machine Learning and AI/HousingData.csv")
print(Variable1.info())
X = Variable1[["ZN", "INDUS", "RM", "AGE", "TAX"]]
Y = Variable1[["MEDV"]]

from sklearn.model_selection import train_test_split as train
Xtrain, Xtest, Ytrain, Ytest = train(X, Y, test_size = 0.15, random_state = 24)
from sklearn.linear_model import LinearRegression
Sketch = LinearRegression()
print(Xtrain)
print(Ytrain)
Sketch.fit(Xtrain, Ytrain)
print(Sketch.coef_)
print(Sketch.intercept_)
VariableY = Sketch.predict(Xtest)

from sklearn.metrics import root_mean_squared_error
X7 = root_mean_squared_error(Ytest, VariableY)
print(X7)
