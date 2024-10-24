"""

    --------------------------------Machine Learning - Confusion Matrix------------------------------

    
    It is a table that is used in classification problems to assess where errors in the model were made.
    
    The rows represent the actual classes the outcomes should have been. While the columns represent 
    the predictions we have made. Using this table it is easy to see which predictions are wrong.
    
    

"""

# create confusion matrix

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

actual = np.random.binomial(1, .9, size=1000)
predicted = np.random.binomial(1, .9, size=1000)

confusion_matrix = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0,1])

cm_display.plot()
plt.show()

"""

    -----------------------------------------Results Explained-----------------------------------------
    
    The Confusion Matrix created has four different quadrants:
        True Negative (Top-Left Quadrant)
        False Positive (Top-Right Quadrant)
        False Negative (Bottom-Left Quadrant)
        True Positive (Bottom-Right Quadrant)
    
    True means that the values were accurately predicted, False means that there was an error or wrong prediction.
    
    
    The matrix provides us with many useful metrics that help us to evaluate out classification model.

    The different measures include: Accuracy, Precision, Sensitivity (Recall), Specificity, and the F-score, explained below.

"""




"""


    -----------------------------------------------Accuracy---------------------------------------
    
    Accuracy measures how often the model is correct.
    
    How to Calculate

                    (True Positive + True Negative) / Total Predictions


"""

import numpy as np
from sklearn import metrics

actual = np.random.binomial(1, .9, size=1000)
predicted = np.random.binomial(1, .9, size=1000)

Accuracy = metrics.accuracy_score(actual, predicted)

print(Accuracy)




"""

    ----------------------------------------Precision---------------------------------------------
    
    Of the positives predicted, what percentage is truly positive?

    How to Calculate
                    
                    True Positive / (True Positive + False Positive)

    
    Precision does not evaluate the correctly predicted negative cases:


"""

import numpy as np
from sklearn import metrics

actual = np.random.binomial(1, .9, size=1000)
predicted = np.random.binomial(1, .9, size=1000)

Precision = metrics.precision_score(actual, predicted)

print(Precision)




"""


------------------------------------------Sensitivity (Recall)---------------------------------------

    Of all the positive cases, what percentage are predicted positive?

    Sensitivity (sometimes called Recall) measures how good the model is at predicting positives.

    This means it looks at true positives and false negatives (which are positives that have been incorrectly predicted as negative).

    How to Calculate
                
                True Positive / (True Positive + False Negative)

    
    
    
    Sensitivity is good at understanding how well the model predicts something is positive:


"""


import numpy as np
from sklearn import metrics

actual = np.random.binomial(1, .9, size=1000)
predicted = np.random.binomial(1, .9, size=1000)

Sensitivity_recall = metrics.recall_score(actual, predicted)

print(Sensitivity_recall)




"""
-----------------------------------------------Specificity---------------------------------------- 

    How well the model is at prediciting negative results?

    Specificity is similar to sensitivity, but looks at it from the persepctive of negative results.

        How to Calculate
                
                    True Negative / (True Negative + False Positive)




Since it is just the opposite of Recall, we use the recall_score function, taking the opposite position label:


"""

import numpy as np
from sklearn import metrics

actual = np.random.binomial(1, .9, size=1000)
predicted = np.random.binomial(1, .9, size=1000)

Specificity = metrics.recall_score(actual, predicted, pos_label=0)

print(Specificity)


"""


----------------------------------------------- F-score----------------------------------------------


    F-score is the "harmonic mean" of precision and sensitivity.

    It considers both false positive and false negative cases and is good for imbalanced datasets.

            How to Calculate
                        
                        
                        2 * ((Precision * Sensitivity) / (Precision + Sensitivity))

    
    
    This score does not take into consideration the True Negative values:

        Example
                
                F1_score = metrics.f1_score(actual, predicted)


"""

import numpy as np
from sklearn import metrics

actual = np.random.binomial(1, .9, size=1000)
predicted = np.random.binomial(1, .9, size=1000)


F1_score = metrics.f1_score(actual, predicted)

print(F1_score)



# metrics
print(f"Accuracy: {Accuracy}, Precision: {Precision}, Sensitivity_Recall: {Sensitivity_recall}, Specificity: {Specificity}, F1_Score: {F1_score}")