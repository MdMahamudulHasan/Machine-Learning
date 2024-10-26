"""
    ----------------------------------------------------Hierarchical Clustering---------------------------------------------
    
    Hierarchical clustering is an unsupervised learning method for clustering data points. The algorithm builds clusters by 
    measuring the dissimilarities between data. Unsupervised learning means that a model does not have to be trained, and we 
    do not need a "target" variable. This method can be used on any data to visualize and interpret the relationship between 
    individual data points.

    Here we will use hierarchical clustering to group data points and visualize the clusters using both a dendrogram and 
    scatter plot.


    
    We will use Agglomerative Clustering, a type of hierarchical clustering that follows a bottom up approach. We begin by 
    treating each data point as its own cluster. Then, we join clusters together that have the shortest distance between 
    them to create larger clusters. This step is repeated until one large cluster is formed containing all of the data points.

    
    Hierarchical clustering requires us to decide on both a distance and linkage method. We will use euclidean distance and 
    the Ward linkage method, which attempts to minimize the variance between clusters.
"""