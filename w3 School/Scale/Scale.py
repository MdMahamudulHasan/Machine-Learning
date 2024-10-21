"""
    Scale Features
    
    
    When your data has different values, and even different measurement units, it can be difficult to compare them. What is kilograms compared to meters? Or altitude compared to time?
    
    
    The answer to this problem is scaling. We can scale data into new values that are easier to compare.
    
    There are different methods for scaling data, in this tutorial we will use a method called standardization.
    
            z = (x - u) / s
        
        ->  z is the new value,
        ->  x is the original value
        ->  u is the mean
        ->  s is the standard deviation.
    
    
    You do not have to do this manually, the Python sklearn module has a method called "StandardScaler()" which returns a Scaler object with methods for transforming data sets.



"""



#Scale all values in the Weight and Volume columns:

import pandas as pd 
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

df = pd.read_csv("G:/Machine Learning/w3 School/Scale/data.csv")

X = df[['Weight', 'Volume']]

scaledX = scale.fit_transform(X)

print(scaledX)



"""

    Predict CO2 Values
    
    The task in the Multiple Regression chapter was to predict the CO2 emission from a car when you only knew its weight and volume.
    
    When the data set is scaled, you will have to use the scale when you predict values:


"""


# Predict the CO2 emission from a 1.3 liter car that weighs 2300 kilograms:

import pandas as pd 
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()



df = pd.read_csv("G:/Machine Learning/w3 School/Scale/data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)