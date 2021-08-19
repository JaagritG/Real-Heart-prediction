import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("heart.csv")
#use required features
cdf = df[['age','cp','chol','target']]
'''
cp = chest pain amt
trestbps = resting blood pressure
chol = cholestrol amount
ca = number of major vessels
thalach = max heart rate
target = chance of heart attack
'''

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :3] # ':3' means no. of columns from the ones in cdf 
y = cdf.iloc[:, -1] # pretty sure -1 means last column
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.


pickle.dump(regressor, open('model.pkl','wb'))

#testing
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[63, 3,233]]))