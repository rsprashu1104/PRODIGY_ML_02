# K-Means Clustering ReadMe

## Overview

This README file provides instructions and explanations for performing K-means clustering on a customer dataset using Python. The script loads the data, preprocesses it, and applies the K-means clustering algorithm to segment the customers based on their age, annual income, and spending score.

## Prerequisites

Ensure you have the following Python libraries installed:

- pandas
- scikit-learn
- matplotlib
- seaborn

You can install these libraries using pip if they are not already installed:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Dataset

The dataset is expected to be in CSV format, named `ml_02.csv`, and should contain the following columns:

- `CustomerID`: Unique identifier for each customer.
- `Gender`: Gender of the customer (Male/Female).
- `Age`: Age of the customer.
- `Annual Income (k$)`: Annual income of the customer in thousand dollars.
- `Spending Score (1-100)`: Spending score assigned to the customer (1-100).

## Steps

1. **Load the Dataset**: Read the CSV file into a pandas DataFrame.
2. **Data Preprocessing**:
    - Convert the `Gender` column to numerical values (0 for Male, 1 for Female).
    - Drop the `CustomerID` column as it is not needed for clustering.
    - Check for and handle any missing values.
3. **Feature Scaling**: Standardize the features using `StandardScaler`.
4. **Elbow Method**: Determine the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) for different values of k.
5. **K-means Clustering**: Apply the K-means algorithm with the chosen number of clusters.
6. **Visualization**: Visualize the clusters and their centroids.

## Code

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('ml_02.csv')

# Convert 'Gender' to numerical values
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Drop 'CustomerID' as it's not needed for clustering
df = df.drop(columns=['CustomerID'])

# Check for null values
print(df.isnull().sum())

# Display the first few rows of the dataset
print(df.head())

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Choose the number of clusters
k = 5

# Initialize KMeans model
kmeans = KMeans(n_clusters=k)

# Fit the model to the data
kmeans.fit(scaled_features)

# Get the cluster centers and labels for each data point
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Add cluster labels to the original dataset
df['Cluster'] = labels

# Visualize the clusters
plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {i}')

# Plot the centroids
plt.scatter(centroids[:, 1], centroids[:, 2], s=300, c='red', marker='*', label='Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-means Clustering')
plt.legend()
plt.show()
```

## Explanation

- **Data Loading and Preprocessing**: The dataset is loaded and the `Gender` column is converted to numerical values. The `CustomerID` column is dropped as it is not required for clustering. The data is checked for any missing values.
- **Feature Scaling**: Features are standardized using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1.
- **Elbow Method**: The within-cluster sum of squares (WCSS) is calculated for k values from 1 to 10 and plotted to identify the optimal number of clusters.
- **K-means Clustering**: The K-means algorithm is applied with the chosen number of clusters (k = 5). The cluster labels are added to the original dataset.
- **Visualization**: The clusters are visualized using a scatter plot, with different colors representing different clusters and red stars indicating the centroids.

## Conclusion

This script performs K-means clustering on a customer dataset to segment customers based on their age, annual income, and spending score. The optimal number of clusters is determined using the Elbow Method, and the resulting clusters are visualized in a scatter plot.
