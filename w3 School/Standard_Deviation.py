"""

    What is Standard Deviation?
    
    Standard deviation is a number that describes how spread out the values are.
    
    
    A low standard deviation means that most of the numbers are close to the mean (average) value.
    A high standard deviation means that the values are spread out over a wider range.
    
            Example: This time we have registered the speed of 7 cars:
                speed = [86,87,88,86,87,85,86]
                The standard deviation is: 0.9
            Meaning that most of the values are within the range of 0.9 from the mean value, which is 86.4.
            
            
            Let us do the same with a selection of numbers with a wider range:
                speed = [32,111,138,28,59,77,97]
                The standard deviation is: 37.85
            Meaning that most of the values are within the range of 37.85 from the mean value, which is 77.4.

"""

#Use the NumPy "std()" method to find the standard deviation:
import numpy as np

speed = [86,87,88,86,87,85,86]

x = np.std(speed)

print(x)



import numpy as np

speed = [32,111,138,28,59,77,97]

x = np.std(speed)

print(x)


"""

    ------------------------------------------------------Variance---------------------------------------------------------
    
    Variance is another number that indicates how spread out the values are.
    
    
    In fact, if you take the square root of the variance, you get the standard deviation!
    Or the other way around, if you multiply the standard deviation by itself, you get the variance!
    

"""

#Use the NumPy "var()" method to find the variance:
import numpy as np 

speed = [32,111,138,28,59,77,97]

x = np.var(speed)

print(x)


# Use the NumPy "std()" method to find the standard deviation:

import numpy

speed = [32,111,138,28,59,77,97]

x = numpy.std(speed)

print(x)


"""
    Symbols-----------------------
    
    Standard Deviation is often represented by the symbol Sigma: σ
    Variance is often represented by the symbol Sigma Squared: σ2

"""