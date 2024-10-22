"""

    Machine Learning - Train/Test
    
    Evaluate Your Model
        In Machine Learning we create models to predict the outcome of certain events, like in the previous chapter where we predicted the CO2 emission of a car when we knew the weight and engine size.

        To measure if the model is good enough, we can use a method called Train/Test.


    What is Train/Test
        Train/Test is a method to measure the accuracy of your model.

        It is called Train/Test because you split the data set into two sets: a training set and a testing set.
                        
                        80% for training, and 20% for testing.

                        
                        
                        You train the model using the training set.

                        You test the model using the testing set.

                        
                        
                        Train the model means create the model.

                        Test the model means test the accuracy of the model.
    
    
"""

# start with a data set
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100) /x #here ues x for perfect data set

plt.scatter(x, y)
plt.show()

#The x axis represents the number of minutes before making a purchase.
#The y axis represents the amount of money spent on the purchase.



"""

    Split Into Train/Test
    
    
    The training set should be a random selection of 80% of the original data.

    The testing set should be the remaining 20%.

            train_x = x[:80]
            train_y = y[:80]

            test_x = x[80:]
            test_y = y[80:]

"""


# Display the training set
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100)/x

train_x = x[ :80]
train_y = y[ :80]

test_x = x[80: ]
test_y = y[80: ]

plt.scatter(train_x, train_y)
plt.show()



# Display Test Data set
plt.scatter(test_x, test_y)
plt.show()


"""

    Fit the Data Set
    
    
    What does the data set look like? In my opinion I think the best fit would be a polynomial regression, so let us draw a line of polynomial regression.
    
    To draw a line through the data points, we use the plot() method of the matplotlib module:


"""



# Draw a polynomial regression line through the data points:

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)


x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100)/x

train_x = x[ :80]
train_y = y[ :80]

test_x = x[80: ]
test_y = y[80: ]

myModel = np.poly1d(np.polyfit(train_x, train_y, 4))

myLine = np.linspace(0, 6, 100)
# perfect -> myLine = np.linspace(min(train_x), max(train_x), 100)

plt.scatter(train_x, train_y)
plt.plot(myLine, myModel(myLine))
plt.show()



"""
what about the R-squared score? The R-squared score is a good indicator of how well my data set is fitting the model.

"""




"""
    R2
    
    It measures the relationship between the x axis and the y axis, and the value ranges from 0 to 1, where 0 means no relationship, and 1 means totally related.

    The sklearn module has a method called r2_score() that will help us find this relationship.

    In this case we would like to measure the relationship between the minutes a customer stays in the shop and how much money they spend.




"""

# How well does my training data fit in a polynomial regression
import numpy as np
from sklearn.metrics import r2_score
np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100)/x

train_x = x[ :80]
train_y = y[ :80]

test_x = x[80: ]
test_y = x[80: ]

myModel = np.poly1d(np.polyfit(train_x, train_y, 4))

r2 =r2_score(train_y, myModel(train_x))

print(r2)

"""The result close to 0.799 shows that there is a OK relationship."""



"""

    Bring in the Testing Set
    
    Now we have made a model that is OK, at least when it comes to training data.

    Now we want to test the model with the testing data as well, to see if gives us the same result.

"""

# Let us find the R2 score when using testing data:
import numpy as np
from sklearn.metrics import r2_score
np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100)/x

train_x = x[ :80]
train_y = y[ :80]

test_x = x[80: ]
test_y = y[80: ]

myModel = np.poly1d(np.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, myModel(test_x))

print(r2)

"""The result 0.809 shows that the model fits the testing set as well, and we are confident that we can use the model to predict future values."""


"""

    Predict Values
    
    Now that we have established that our model is OK, we can start predicting new values.
    

"""


# How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?
import numpy as np
from sklearn.metrics import r2_score
np.random.seed(2)

x = np.random.normal(3, 1, 100)
y = np.random.normal(150, 40, 100)/x

train_x = x[ :80]
train_y = y[ :80]

test_x = x[80: ]
test_y = y[80: ]

myModel = np.poly1d(np.polyfit(train_x, train_y, 4))

price = myModel(5)

print(price)


#plotting the prediction value
plt.scatter(train_x, train_y, color='blue', label='Train and Testing Data point')
plt.plot(myLine, myModel(myLine), color='red', label='Polynomial Fit')
plt.hlines(price, xmin=min(train_x), xmax=5, colors='yellow', label='Prediction Point line')
plt.vlines(5, ymin=min(train_y), ymax=price, colors='yellow')


plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Prediction Point Using Testing and Training Method')

plt.legend()
plt.show()
