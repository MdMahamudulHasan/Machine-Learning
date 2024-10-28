"""

    -----------------------------------Preprocessing - Categorical Data-------------------------------

    
    
    When your data has categories represented by strings, it will be difficult to use them to train 
    machine learning models which often only accepts numeric data.

    Instead of ignoring the categorical data and excluding the information from our model, you can 
    transform the data so it can be used in your models.


"""

import pandas as pd 

cars = pd.read_csv("G:/Machine Learning/w3 School/Multiple Regression/data.csv")
print(cars.to_string())






"""

    ---------------------------------------------One Hot Encoding-------------------------------------
    
    We cannot make use of the Car or Model column in our data since they are not numeric. A linear 
    relationship between a categorical variable, Car or Model, and a numeric variable, CO2, cannot be determined.

    To fix this issue, we must have a numeric representation of the categorical variable. One way to do 
    this is to have a column representing each group in the category.

    For each column, the values will be 1 or 0 where 1 represents the inclusion of the group and 0 
    represents the exclusion. This transformation is called one hot encoding.

    the Python Pandas module has a function that called get_dummies() which does one hot encoding.


"""


# one Hot Encode the car column


import pandas as pd 

cars = pd.read_csv("G:/Machine Learning/w3 School/Multiple Regression/data.csv")
ohe_cars = pd.get_dummies(cars[['Car']], dtype=int)

print(ohe_cars.to_string())



"""
----------------------------------------------- Predict CO2---------------------------------------

    We can use this additional information alongside the volume and weight to predict CO2

    To combine the information, we can use the concat() function from pandas.


"""

import pandas as pd 
from sklearn import linear_model

cars = pd.read_csv("G:/Machine Learning/w3 School/Multiple Regression/data.csv")
ohe_cars = pd.get_dummies(cars[['Car']])

X = pd.concat([cars[['Volume', 'Weight']], ohe_cars], axis=1)
y = cars['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

##predict the CO2 emission of a Volvo where the weight is 2300kg, and the volume is 1300cm3:
input_data =pd.DataFrame([[2300, 1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]], columns=X.columns)

predictedCO2 = regr.predict(input_data)

print(predictedCO2)



"""

------------------------------------------------ Dummifying------------------------------------------
    
    
    It is not necessary to create one column for each group in your category. The information can be 
    retained using 1 column less than the number of groups you have.


"""

import pandas as pd 

colors = pd.DataFrame({'color': ['blue', 'red']})

print(colors)

"""
    What if you have more than 2 groups? How can the multiple groups be represented by 1 less column?

    Let's say we have three colors this time, red, blue and green. When we get_dummies while dropping 
    the first column, we get the following table.

"""
import pandas as pd 

colors = pd.DataFrame({'color': ['blue', 'red']})
dummies = pd.get_dummies(colors, drop_first=True, dtype=int)

print(dummies)


# when multiple group have

import pandas as pd 

colors = pd.DataFrame({'color': ['blue', 'red', 'green']})
dummies = pd.get_dummies(colors, drop_first=True, dtype=int)
dummies['color'] = colors['color']

print(dummies)




import pandas as pd 

colors = pd.DataFrame({'color': ['blue', 'red', 'green']})
dummies = pd.get_dummies(colors,  dtype=int)
dummies['color'] = colors['color']

print(dummies)