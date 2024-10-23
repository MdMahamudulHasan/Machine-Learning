"""

    ----------------------------------------------------------------------Decision Tree-------------------------------------
    
    
    A Decision Tree is a Flow Chart, and can help you make decisions based on previous experience.
    
    
    In the example, a person will try to decide if he/she should go to a comedy show or not.
    

"""
#First, read the dataset with pandas:
import pandas as pd 

df = pd.read_csv("G:/Machine Learning/w3 School/Decision Tree/data.csv")

print (df.head())

"""To make a decision tree, all data has to be numerical."""
"""We have to convert the non numerical columns 'Nationality' and 'Go' into numerical values."""


#Change string values into numerical values:
import pandas as pd 

d = {'UK': 0, 'USA': 1, 'N':2}
df['Nationality'] = df['Nationality'].map(d)

d = {'NO': 0, 'YES': 1}
df['Go'] = df['Go'].map(d)

print(df.head())

"""

    Then we have to separate the feature columns from the target column.

    The feature columns are the columns that we try to predict from, and the target column is the column with the values we try to predict.


"""

# X is the feature columns, y is the target column:

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

print(X)
print(y)

"""

    Now we can create the actual decision tree, fit it with our details. Start by importing the modules we need:


"""
#Create and display a Decision Tree:
import pandas as pd 
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("G:/Machine Learning/w3 School/Decision Tree/data.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality']= df['Nationality'].map(d)

d= {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

#make our compiler able to draw
import sys
import matplotlib
matplotlib.use('Agg')


plt.savefig(sys.stdout.buffer)
sys.stdout.flush()





plt.figure(figsize=(12, 8))  
tree.plot_tree(dtree, feature_names=features, filled=True, rounded=True, class_names=['NO', 'YES'])

# Display the plot
plt.show()


"""

    ---------------------------------------------------------Predict Values-------------------------------------------------
    We can use the Decision Tree to predict new values.

        Example: Should I go see a show starring a 40 years old American comedian, with 10 years of experience, and a comedy ranking of 7?


"""

#Use predict() method to predict new values:
import pandas as pd 
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("G:/Machine Learning/w3 School/Decision Tree/data.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality']= df['Nationality'].map(d)

d= {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)


# Predict with new data
data = pd.DataFrame([[40, 10, 6, 1]], columns=features)
print(dtree.predict(data)) 
print("[1] means 'GO'\n[0] means 'NO'\n")


# Another prediction
data2 = pd.DataFrame([[40, 10, 7, 1]], columns=features)
print(dtree.predict(data2)) 
print("[1] means 'GO'\n[0] means 'NO'\n")




#--------------------------------------------------Visualizing Feature Importance-----------------------------------
import matplotlib
matplotlib.use('TkAgg')  # Choose an interactive backend
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("G:/Machine Learning/w3 School/Decision Tree/data.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Plot feature importance
importance = dtree.feature_importances_
plt.barh(features, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in Decision Tree')
plt.show()



# ---------------------------------------------Visualizing the Decision Boundaries-----------------------------

import numpy as np
from matplotlib.colors import ListedColormap

# Select two features to visualize decision boundary
X_two_features = df[['Age', 'Experience']].values

dtree = DecisionTreeClassifier().fit(X_two_features, y)

x_min, x_max = X_two_features[:, 0].min() - 1, X_two_features[:, 0].max() + 1
y_min, y_max = X_two_features[:, 1].min() - 1, X_two_features[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = dtree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ['red', 'blue']

plt.contourf(xx, yy, Z, cmap=cmap_light)

# Plot the points as well
plt.scatter(X_two_features[:, 0], X_two_features[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor='k', s=20)
plt.xlabel('Age')
plt.ylabel('Experience')
plt.title('Decision Boundary')
plt.show()






#----------------------------------------------Confusion Matrix for Model Evaluation:-------------------------------

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Close all open figures (suppress the warning)
plt.close('all')

# Choose an interactive backend (set once)
matplotlib.use('TkAgg')

# Load the data
df = pd.read_csv("G:/Machine Learning/w3 School/Decision Tree/data.csv")

# Map non-numerical columns to numerical values
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

# Define the features and target variable
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

# Fit the Decision Tree model
dtree = DecisionTreeClassifier()
dtree.fit(X, y)

# Make a prediction for new data using a DataFrame
new_data = pd.DataFrame([[40, 10, 7, 1]], columns=features)  # Ensure proper naming
y_pred_new = dtree.predict(new_data)

# Make predictions using the same dataset for confusion matrix
y_pred = dtree.predict(X)

# Generate confusion matrix
cm = confusion_matrix(y, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NO', 'GO'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Print the predictions for new data
print("Prediction for new data [40, 10, 7, 1]:", y_pred_new[0])

# Optionally print the confusion matrix
print("Confusion Matrix:\n", cm)