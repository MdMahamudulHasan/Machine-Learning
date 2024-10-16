"""

    -------------------------------------------------------Scatter Plot-------------------------------------------------
    
    A scatter plot is a diagram where each value in the data set is represented by a dot.
    
    The Matplotlib module has a method for drawing scatter plots, it needs two arrays of the same length, one for the values of the x-axis, and one for the values of the y-axis:
    
        x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
        y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
    
    The "x" array represents the age of each car.
    The "y" array represents the speed of each car.

"""

# Use the "scatter()" method to draw a scatter plot diagram:
import numpy as np
import matplotlib.pyplot as plt

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()



"""

    ---------------------------------------------------Random Data Distributions-----------------------------------------
    
    In Machine Learning the data sets can contain thousands-, or even millions, of values.
    
    You might not have real world data when you are testing an algorithm, you might have to use randomly generated values.

"""

#A scatter plot with 1000 dots:
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(5.0, 1.0, 1000)
y = np.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()