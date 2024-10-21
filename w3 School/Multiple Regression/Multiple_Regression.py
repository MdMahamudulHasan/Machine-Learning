"""
    -------------------------------------Multiple Regression-------------------------------------------
    
    Multiple regression is like linear regression, but with more than one independent value, meaning 
    that we try to predict a value based on two or more variables.
    
    
    
    
    In Python we have modules that will do the work for us. Start by importing the Pandas module.
    
                import pandas
    
    
    The Pandas module allows us to read csv files and return a DataFrame object.
    
                df = pandas.read_csv("data.csv")

    Then make a list of the independent values and call this variable X.
    Put the dependent values in a variable called y.
    
    X = df[['Weight', 'Volume']]
    y = df['CO2']
    
    "" It is common to name the list of independent values with a upper case X, and the list of 
    dependent values with a lower case y.""
    
    We will use some methods from the sklearn module, so we will have to import that module as well:

                from sklearn import linear_model

    From the sklearn module we will use the LinearRegression() method to create a linear regression object.

    This object has a method called fit() that takes the independent and dependent values as parameters 
    and fills the regression object with data that describes the relationship:

                regr = linear_model.LinearRegression()
                regr.fit(X, y)
"""


# see the whole example in action
import pandas as pd 
from sklearn import linear_model

df = pd.read_csv("G:/Machine Learning/w3 School/Multiple Regression/data.csv")

#print(df.head())

X = df[['Weight','Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
# Predicting CO2 emission with a DataFrame having proper feature names
new_data = pd.DataFrame({'Weight': [2300], 'Volume': [1300]})
predictedCO2 = regr.predict(new_data)

print(predictedCO2)



import matplotlib.pyplot as plt
# Plot Weight vs CO2
plt.scatter(df['Weight'], y, color='blue', label='Actual Data: Weight vs CO2')

# Plot the predicted point
plt.scatter(new_data['Weight'], predictedCO2, color='red', label='Predicted CO2 (Weight=2300, Volume=1300)', marker='X', s=100)

# Add labels and title
plt.xlabel('Weight (g)')
plt.ylabel('CO2 Emissions')
plt.title('Weight vs CO2 Emissions with Prediction')

# Show legend
plt.legend()

# Display the plot
plt.show()





# Plot Volume vs CO2
plt.scatter(df['Volume'], y, color='green', label='Actual Data: Volume vs CO2')

# Plot the predicted point on Volume vs CO2 graph
plt.scatter(new_data['Volume'], predictedCO2, color='red', label='Predicted CO2 (Weight=2300, Volume=1300)', marker='X', s=100)

# Add labels and title
plt.xlabel('Volume (ccm)')
plt.ylabel('CO2 Emissions')
plt.title('Volume vs CO2 Emissions with Prediction')

# Show legend
plt.legend()

# Display the plot
plt.show()







# Plot Weight vs CO2
plt.scatter(df['Weight'], y, color='blue', label='Actual Data: Weight vs CO2', alpha=0.6)

# Plot Volume vs CO2 using the same y-axis
plt.scatter(df['Volume'], y, color='green', label='Actual Data: Volume vs CO2', alpha=0.6)

# Plot the predicted point
plt.scatter(new_data['Weight'], predictedCO2, color='red', label='Predicted CO2 (Weight=2300, Volume=1300)', marker='X', s=100)

# Add labels and title
plt.xlabel('Weight (g) and Volume (ccm)')
plt.ylabel('CO2 Emissions')
plt.title('Weight and Volume vs CO2 Emissions with Prediction')

# Show legend
plt.legend()

# Display the plot
plt.show()





"""

    ----------------------------------------------Coefficient------------------------------------------
    
    
    The coefficient is a factor that describes the relationship with an unknown variable.
            Example: if x is a variable, then 2x is x two times. x is the unknown variable, and the number 2 is the coefficient.
    
    
    In this case, we can ask for the coefficient value of weight against CO2, and for volume against 
    CO2. The answer(s) we get tells us what would happen if we increase, or decrease, one of the 
    independent values.
    

"""


#Print the coefficient values of the regression object:

import pandas as pd 
from sklearn import linear_model


df = pd.read_csv("G:/Machine Learning/w3 School/Multiple Regression/data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)


"""


    The result array represents the coefficient values of weight and volume.

            Weight: 0.00755095
            Volume: 0.00780526

    These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.

    And if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.

    I think that is a fair guess, but let test it!

    We have already predicted that if a car with a 1300cm3 engine weighs 2300kg, the CO2 emission will be approximately 107g.

    What if we increase the weight with 1000kg?


"""




#Copy the example from before, but change the weight from 2300 to 3300:

import pandas as pd 
from sklearn import linear_model

df = pd.read_csv("G:/Machine Learning/w3 School/Multiple Regression/data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

new_data = pd.DataFrame({'Weight': [3300], 'Volume': [1300]})
predictedCO2 = regr.predict(new_data)

print(predictedCO2)


"""
    We have predicted that a car with 1.3 liter engine, and a weight of 3300 kg, will release approximately 115 grams of CO2 for every kilometer it drives.

    Which shows that the coefficient of 0.00755095 is correct:

        107.2087328 + (1000 * 0.00755095) = 114.75968


"""