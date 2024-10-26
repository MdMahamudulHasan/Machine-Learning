"""

--------------------------------------Logistic Regression--------------------------------------------
    
    
    Logistic Regression aims to solve classification problems. It does this by predicting categorical outcomes, unlike linear regression that predicts a continuous outcome.
    
    In the simplest case there are two outcomes, which is called binomial, an example of which is 
    predicting if a tumor is malignant or benign.
    
    Other cases have more than two outcomes to classify, in this case it is called multinomial. A 
    common example for multinomial logistic regression would be predicting the class of an iris flower 
    between 3 different species.
    
    here wil be use basic logistic regression to predict a binomial variable. This means it has only 
    two possible outcomes.

    
    



"""

#predict if tumor is cancerous where the size is 3.46mm:

import numpy as np
from sklearn import linear_model

# Reshaped for logistic Regression
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logR = linear_model.LogisticRegression()
logR.fit(X, y)

#predict if tumor is cancerous where the size is 3.46mm:
predicted = logR.predict(np.array([3.46]).reshape(-1, 1))
print(predicted)

"""

We have predicted that a tumor with a size of 3.46mm will not be cancerous.


"""





"""


-------------------------------------Coefficient---------------------------------------------------
    
    In logistic regression the coefficient is the expected change in log-odds of having the outcome per unit change in X.

    This does not have the most intuitive understanding so let's use it to create something that makes more sense, odds.


"""

# coefficient
import numpy as np
from sklearn import linear_model

#Reshaped for Logistic function.
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])


logR = linear_model.LogisticRegression()
logR.fit(X, y)

log_odds = logR.coef_
odds = np.exp(log_odds)

print(odds)



"""


-------------------------------------Probability--------------------------------------------------------
    
    The coefficient and intercept values can be used to find the probability that each tumor is cancerous.

    Create a function that uses the model's coefficient and intercept values to return a new value. This new value represents probability that the given observation is a tumor:

            def logit2prob(logr,x):
                log_odds = logr.coef_ * x + logr.intercept_
                odds = numpy.exp(log_odds)
                probability = odds / (1 + odds)
                return(probability)

"""


#Let us now use the function with what we have learned to find out the probability that each tumor is cancerous.

import numpy as np
from sklearn import linear_model

#Reshaped for Logistic function.
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logR = linear_model.LogisticRegression()
logR.fit(X, y)

def logistic2prob( logR, X):
    log_odds = logR.coef_ * X + logR.intercept_
    odds = np.exp(log_odds)
    probability = odds/(1+odds)
    return(probability)

print(logistic2prob(logR, X))

""" 

------------------------------Results Explained---------------------------------------------------

    3.78 0.61 The probability that a tumor with the size 3.78cm is cancerous is 61%.

    2.44 0.19 The probability that a tumor with the size 2.44cm is cancerous is 19%.

    2.09 0.13 The probability that a tumor with the size 2.09cm is cancerous is 13%.

"""