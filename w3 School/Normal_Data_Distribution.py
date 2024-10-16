"""

    --------------------------------------Normal Data Distribution---------------------------------
    
    In probability theory this kind of data distribution is known as the normal data distribution, or the Gaussian data distribution, after the mathematician Carl Friedrich Gauss who came up with the formula of this data distribution.
    
    
    
"""
# A typical normal data distribution:
import numpy as np 
import matplotlib.pyplot as plt

x = np.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

"""A normal distribution graph is also known as the bell curve because of it's characteristic shape of a bell."""


