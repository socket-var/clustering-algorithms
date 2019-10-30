# CSE 601 - Project 2: Clustering Algorithms

# Team members:

**Saketh Varma Pericherla** - **sakethva**

**Arvind Thirumurugan** - **athirumu**

**Vijay Jagannathan** - **vijayjag**

# K-means:

## Algorithm Description:

## Result Visualization:

## Result Evaluation:

# Hierarchical Agglomerative clustering with Min approach:

## Algorithm Description:

## Result Visualization:

## Result Evaluation:

# DBSCAN:

- Density Based Spacial Clustering of Applications with Noise or DBSCAN is a clustering algorithm that seperates data points based on the density of the region they belong to.
- Clusters are defined as dense regions in the data space, separated from regions of lower density.
- The size of the region is defined by the  &#949;-Neighborhood which gives the region within a radius of &#949; from a data point.
- Before jumping into the algorithm, some of the key concepts are described below:
- **Core point** - A point is a core point if it has more than a specified number of points (MinPts) within Eps—These are points that are at the interior of a cluster.
- **Border point** - A border point has fewer than MinPts within Eps, but is in the neighborhood of a core point.
- **Noise point** - A noise point is any point that is not a core point nor a border point.
- **Directly density-reachable** - An object q is directly density-reachable from object p if p is a core object and q is in p’s &#949;-neighborhood.



## Algorithm:

For each point in the dataset that is not yet classified do the following steps:
- Check if the point is a core point or not.
- If the point is not a core point label it as a noise point
- If the point is indeed a core point, collect all the points that are density reachable from the current point and label them as a new cluster
- Repeat the above steps until all the points are visited and labeled into a cluser or as noise.

## Implementation:

1. The parameters &#949; and MinPts are specified when running the program. We choose a small epsilon value typically between 0.5 to 1.5 and MinPts value as 3 for optimal results.
2. Using the 'regionQuery' function find the neighbors of the point using &#949; and MinPts and euclidean distance, for each unvisited point in the matrix.

3. If the number of neighbors is more than the minimum number of points, classify the point to the corresponding cluster id.
4. If the number of neighbors is less than the minimum number of points, classify the label of that
point to be -1.
5. The 'expandCluster' function is used to assign every unvisited point
in the neighborhood of point to the same cluster id. The point's neighbors are calculated by 'regionQuery' and then added on to the existing neighboring points.

## Result Visualization:

### **iyer.txt:**

Rand value is: 0.6526755683922646

Jaccard value is: 0.2835567491646023

![](./results/dbscan_iyer.PNG)

### **cho.txt:**

Rand value is: 0.5664715831297484

Jaccard value is: 0.2037425112793077


![](./results/dbscan_cho.PNG)

## Result Evaluation:

 - The time complexity of this algorithm is O(n<sup>2</sup>), so it is reasonably fast when compared to algorithms like HAC and Spectral Clustering.
- If dataset is too large small clusters are likely to be labeled as noise
- If dataset is too small, even a small number of closely spaced that are noise or outliers will be incorrectly labeled as clusters.

### **Advantages:**
- DBSCAN can find arbitrarily sized and shaped clusters.
- We don’t need to specify the number of clusters, as opposed to k-means.
- DBSCAN is resistant to outliers, as opposed to k-means.

### **Disadvantages:**
- DBSCAN cannot perform well with data spread across varying densities since &#949; and MinPts are provided as constants for the entire dataset.
- DBSCAN is not completely deterministic. Depending on the order of processing the data, border points reachable from more than one
cluster can be part of different clusters across multiple runs of the same algorithm.


# Guassian Mixture Model:

## Algorithm Description:

## Result Visualization:

## Result Evaluation:

# Spectral Clustering:

## Algorithm Description:

## Result Visualization:

## Result Evaluation:


# Overall Comparison:

Add rand and jaccard values table for all algorithms