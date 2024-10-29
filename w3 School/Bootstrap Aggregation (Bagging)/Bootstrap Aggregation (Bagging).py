"""
    -----------------------------------------------Bootstrap Aggregation (Bagging)----------------------------------------
    
    Methods such as Decision Trees, can be prone to overfitting on the training set which can lead to wrong predictions on new data.

    Bootstrap Aggregation (bagging) is a ensembling method that attempts to resolve overfitting for classification or 
    regression problems. Bagging aims to improve the accuracy and performance of machine learning algorithms. It does this 
    by taking random subsets of an original dataset, with replacement, and fits either a classifier (for classification) or 
    regressor (for regression) to each subset. The predictions for each subset are then aggregated through majority vote for 
    classification or averaging for regression, increasing prediction accuracy.


    
    
    ---------------------------------------------------Evaluating a Base Classifier----------------------------------------

    
    To see how bagging can improve model performance, we must start by evaluating how the base classifier performs on the 
    dataset. If you do not know what decision trees are review the lesson on decision trees before moving forward, as 
    bagging is a continuation of the concept.


"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = datasets.load_wine(as_frame=True)

x = data.data 
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)

dtree = DecisionTreeClassifier(random_state=22)
dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)

train_data_accuracy = accuracy_score(y_true=y_train, y_pred=dtree.predict(x_train))

test_data_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Train data accuracy: ", train_data_accuracy)
print("Test data accuracy: ", test_data_accuracy)



"""


    ----------------------------------------------------Creating a Bagging Classifier---------------------------------------
    
    
    For bagging we need to set the parameter n_estimators, this is the number of base classifiers that our model is going to 
    aggregate together.

    
    For this sample dataset the number of estimators is relatively low, it is often the case that much larger ranges are 
    explored. Hyperparameter tuning is usually done with a grid search, but for now we will use a select set of values for 
    the number of estimators.




"""


# Import the necessary data and evaluate the BaggingClassifier performance.

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

data = datasets.load_wine(as_frame=True)

x = data.data 
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(x , y, test_size=0.25, random_state=22)

estimator_range = [2, 4, 6, 8, 10, 12, 14, 16]

models = []
scores = []

for n_estimators in estimator_range:
    
    # create bagging classifier
    clf = BaggingClassifier(n_estimators =  n_estimators, random_state = 22)
    
    # fit the model 
    clf.fit(x_train, y_train)
    
    # Append the model and score to their respective list
    models.append(clf)
    scores.append(accuracy_score(y_true = y_test, y_pred = clf.predict(x_test)))



# Generate the plot of scores against number of estimators
plt.figure(figsize=(9, 6))
plt.plot(estimator_range, scores)


# adjust labels and font (to make visible)
plt.xlabel("n_estimators", fontsize = 18)
plt.ylabel("score", fontsize = 18)
plt.tick_params(labelsize = 16)

plt.show()



"""
--------------------------------------------- Another Form of Evaluation---------------------------------------------------


    As bootstrapping chooses random subsets of observations to create classifiers, there are observations that are left out 
    in the selection process. These "out-of-bag" observations can then be used to evaluate the model, similarly to that of a 
    test set. Keep in mind, that out-of-bag estimation can overestimate error in binary classification problems and should 
    only be used as a compliment to other metrics.

    We saw in the last exercise that 12 estimators yielded the highest accuracy, so we will use that to create our model. 
    This time setting the parameter oob_score to true to evaluate the model with out-of-bag score.


"""





# Create a model with out-of-bag metric.

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

data = datasets.load_wine(as_frame=True)

x = data.data 
y = data.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 22)

oob_model = BaggingClassifier(n_estimators=12, oob_score = True, random_state=22)
oob_model.fit(x_train, y_train)

print(oob_model.oob_score_)


"""
    Since the samples used in OOB and the test set are different, and the dataset is relatively small, there is a difference 
    in the accuracy. It is rare that they would be exactly the same, again OOB should be used quick means for estimating 
    error, but is not the only evaluation metric.

"""




"""




---------------------------------------Generating Decision Trees from Bagging Classifier-----------------------------------

    As was seen in the Decision Tree lesson, it is possible to graph the decision tree the model created. It is also 
    possible to see the individual decision trees that went into the aggregated classifier. This helps us to gain a more 
    intuitive understanding on how the bagging model arrives at its predictions.

    Note: This is only functional with smaller datasets, where the trees are relatively shallow and narrow making them easy to visualize.

    We will need to import plot_tree function from sklearn.tree. The different trees can be graphed by changing the 
    estimator you wish to visualize.





"""

# Generate Decision Trees from Bagging Classifier

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier

# Load the wine dataset
data = datasets.load_wine(as_frame=True)

# Define features and target
x = data.data 
y = data.target 

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)

# Create a Bagging Classifier (default base estimator is DecisionTreeClassifier)
clf = BaggingClassifier(n_estimators=12, oob_score=True, random_state=22)

# Fit the model
clf.fit(x_train, y_train)

# Plot the first decision tree from the ensemble
plt.figure(figsize=(10, 10))
plot_tree(clf.estimators_[0], feature_names=x.columns, filled=True)  # Accessing the first tree

plt.show()



# Feature Importance Visualization

import pandas as pd
import seaborn as sns

# Calculate feature importances from the first estimator
importances = clf.estimators_[0].feature_importances_
features = x.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort values
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Create a bar plot for feature importance
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from the First Estimator', fontsize=18)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.show()




# Confusion Matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict the test set results
y_pred = clf.predict(x_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix', fontsize=18)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.show()



