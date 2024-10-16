"""

    --------------------------------------------Data Distribution---------------------------------
    
    In the real world, the data sets are much bigger, but it can be difficult to gather real world data, at least at an 
    early stage of a project.
    
    

"""

# Create an array containing 250 random floats between 0 and 5:
import numpy as np

x = np.random.uniform(0, 5, 250)

print(x)

# for integer numbers
x = np.random.randint(1, 5, 250)
print(x)



"""

    ----------------------------------------------Histogram----------------------------------------
    
    To visualize the data set we can draw a histogram with the data we collected.
    We will use the Python module Matplotlib to draw a histogram.

"""

# Draw a histogram
import numpy as np 
import matplotlib.pyplot as plt

x = np.random.uniform(0.0, 5.0, 250)

plt.hist(x, 5)
plt.show()

x = np.random.randint(0, 5, 250)
plt.hist(x, 5)
plt.show()


"""

    --------------------------------------Big Data Distributions----------------------------------
    
    An array containing 250 values is not considered very big, but now you know how to create a random set of values, and by 
    changing the parameters, you can create the data set as big as you want.


"""

# Create an array with 100000 random numbers, and display them using a histogram with 100 bars:
import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0.0, 5.0, 100000)

plt.hist(x, 100)
plt.show()


x = np.random.randint(0, 6, 100000)
plt.hist(x, 100)
plt.show()