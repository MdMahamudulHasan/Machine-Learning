"""

        ----------------------------------Polynomial Regression---------------------------------------
        
        
        If your data points clearly will not fit a linear regression (a straight line through all data 
        points), it might be ideal for polynomial regression.
        
        
        Polynomial regression, like linear regression, uses the relationship between the variables x 
        and y to find the best way to draw a line through the data points.

"""

# start by drawing a scatter plot

import matplotlib.pyplot as plt

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

plt.scatter(x, y)
plt.show()





# Draw Polynomial Regression:

import matplotlib.pyplot as plt
import numpy as np

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

myModel = np.poly1d(np.polyfit(x, y, 3))

myLine = np.linspace(1, 22, 100)

plt.scatter(x, y)
plt.plot(myLine, myModel(myLine))
plt.show()



"""
    NumPy has a method that lets us make a polynomial model:
    
        mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
        
        n the code, numpy.polyfit(x, y, 3) fits a polynomial model to the data points represented by x and y. The third 
        parameter, 3, in this case, represents the degree of the polynomial. Specifically, it indicates that we want to fit 
        a cubic polynomial to the data, which is of the form:
                
                y = ax^3 + bx^2 + cx + d



"""


# generate graph for degree

import numpy as np
import matplotlib.pyplot as plt


# Data points
x = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
y = np.array([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])

# Polynomial fits of different degrees
model_1 = np.poly1d(np.polyfit(x, y, 1)) #Linear
model_2 = np.poly1d(np.polyfit(x, y, 2)) #Quadratic
model_3 = np.poly1d(np.polyfit(x, y, 3)) #Cubic
model_4 = np.poly1d(np.polyfit(x, y, 4)) #Quartic

# values for plotting
x_linspace = np.linspace(min(x), max(x), 100)

plt.scatter(x, y, color = 'blue', label = 'Data points')

#plot for each degree model
plt.plot(x_linspace, model_1(x_linspace), color = 'red', label = 'Degree 1(Linear)')
plt.plot(x_linspace, model_2(x_linspace), color = 'green', label = 'Degree 2(Quadratic)')
plt.plot(x_linspace, model_3(x_linspace), color = 'orange', label = 'Degree 3(Cubic)')
plt.plot(x_linspace, model_4(x_linspace), color = 'purple', label = 'Degree 4(Quartic)')


# Customizing the plot
plt.title("Polynomial Fits of Different Degrees")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.show()



"""

    Deciding the degree of the polynomial (whether to use 3, 4, 5, etc.) depends on a balance between model complexity and 
    generalization. Here's how you can make that decision:
    
    Key Considerations for Deciding the Polynomial Degree:
            1.  Complexity of the Data:
            2.  Avoiding Overfitting:
            3.  Cross-Validation:
            4.  Number of Data Points:
            5.  Domain Knowledge:

    
    General Guidelines:
        ->  Degree 1 (Linear): Use for simple trends (straight-line relationships).
        ->  Degree 2 (Quadratic): Use when you see a parabolic (U-shaped) pattern in the data.
        ->  Degree 3 (Cubic): Often a good balance for moderately complex data with one or two bends.
        ->  Degree 4 or 5 (Higher Degrees): Use if the data shows multiple peaks or turns. However, be cautious of overfitting and validate the model.

"""





"""

    R-Squared
    
    
    It is important to know how well the relationship between the values of the x- and y-axis is, if there are no relationship the polynomial regression can not be used to predict anything.
    
    The relationship is measured with a value called the r-squared.
    
    The r-squared value ranges from 0 to 1, where 0 means no relationship, and 1 means 100% related.


"""

# How well does my data fit in a polynomial regression?

import numpy as np
from sklearn.metrics import r2_score

x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

myModel = np.poly1d(np.polyfit(x, y, 3))

print(r2_score(y, myModel(x)))

"""The result 0.94 shows that there is a very good relationship, and we can use polynomial regression in future predictions."""


