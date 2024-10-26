"""
    --------------------------------Hierarchical Clustering-----------------------------------------
    
    Hierarchical clustering is an unsupervised learning method for clustering data points. The 
    algorithm builds clusters by measuring the dissimilarities between data. Unsupervised learning 
    means that a model does not have to be trained, and we do not need a "target" variable. This method 
    can be used on any data to visualize and interpret the relationship between individual data points.

    Here we will use hierarchical clustering to group data points and visualize the clusters using both a dendrogram and scatter plot.


    
    We will use Agglomerative Clustering, a type of hierarchical clustering that follows a bottom up 
    approach. We begin by treating each data point as its own cluster. Then, we join clusters together 
    that have the shortest distance between them to create larger clusters. This step is repeated until 
    one large cluster is formed containing all of the data points.

    
    Hierarchical clustering requires us to decide on both a distance and linkage method. We will use 
    euclidean distance and the Ward linkage method, which attempts to minimize the variance between clusters.



"""

# start by visualizing some data points

import numpy as np
import matplotlib.pyplot as plt

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

plt.scatter(x, y)
plt.show()



#Now we compute the ward linkage using euclidean distance, and visualize it using a dendrogram:

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]


data = list(zip(x, y))

linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data)

plt.show()




#Here, we do the same thing with Python's scikit-learn library. Then, visualize on a 2-dimensional plot:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

x = [4, 5, 10, 4, 3, 11, 14, 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]


data = list(zip(x, y))

# Correctly initializing AgglomerativeClustering without affinity
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = hierarchical_cluster.fit_predict(data) 

# Now you can plot
plt.scatter(x, y, c=labels)
plt.title('Hierarchical Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()