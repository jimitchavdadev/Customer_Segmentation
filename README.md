# Customer Segmentation using K-Means Clustering

## Project Objective
The objective of this project is to analyze customer data and segment customers based on their purchasing behavior using K-Means clustering. By identifying distinct customer groups, businesses can tailor marketing strategies, enhance customer experience, and improve sales.

## Dataset Information
The dataset used in this project is the **Mall Customers Dataset**, available on Kaggle:
[Download Dataset](https://www.kaggle.com/datasets/shwetabh123/mall-customers)

### Dataset Features:
- **CustomerID**: Unique identifier for each customer (removed for analysis)
- **Gender**: Gender of the customer (Male/Female)
- **Age**: Age of the customer
- **Annual Income (k$)**: Customer's annual income in thousands of dollars
- **Spending Score (1-100)**: A score assigned by the mall based on customer spending behavior

## Theory Behind Customer Segmentation
Customer segmentation is the process of dividing a customer base into distinct groups that share similar characteristics.

### Why Use K-Means Clustering?
K-Means is an unsupervised machine learning algorithm used for clustering data points into **K** clusters. It minimizes the variance within each cluster while maximizing the variance between clusters. The steps of K-Means clustering are:
1. **Initialize K cluster centroids** randomly.
2. **Assign each data point** to the nearest centroid.
3. **Recalculate centroids** based on assigned data points.
4. **Repeat steps 2 and 3** until centroids remain stable or a stopping criterion is met.

## Code Explanation

### 1. Importing Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```
These libraries are used for data manipulation, visualization, and clustering.

### 2. Loading and Preprocessing Data
```python
df = pd.read_csv('Mall_Customers.csv')
df.rename(columns={'Genre':'Gender'}, inplace=True)
df.drop(['CustomerID'], axis=1, inplace=True)
```
- The dataset is loaded using Pandas.
- The column **'Genre'** is renamed to **'Gender'**.
- The **CustomerID** column is dropped as it's not useful for clustering.

### 3. Exploratory Data Analysis (EDA)
#### Checking for Null Values
```python
df.isnull().sum()
```
Ensures there are no missing values in the dataset.

#### Data Visualization
##### Distribution Plots
```python
plt.figure(figsize=(15,6))
n=0
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
  n+=1
  plt.subplot(1,3,n)
  sns.histplot(df[x], bins=20, kde=True)
  plt.title(f'Distribution of {x}')
plt.show()
```
Shows the distribution of Age, Annual Income, and Spending Score.

##### Gender Count Plot
```python
sns.countplot(y='Gender', data=df)
plt.show()
```
Displays the number of male and female customers.

##### Violin Plot for Gender vs Other Variables
```python
sns.violinplot(x='Age', y='Gender', data=df)
sns.violinplot(x='Annual Income (k$)', y='Gender', data=df)
sns.violinplot(x='Spending Score (1-100)', y='Gender', data=df)
plt.show()
```
Shows distribution differences between genders.

### 4. Clustering with K-Means
#### Finding Optimal K Using Elbow Method
```python
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(df[['Age', 'Spending Score (1-100)']])
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```
- **WCSS (Within-Cluster Sum of Squares)** is plotted for different values of K.
- The optimal K is the point where the WCSS curve bends (elbow point).

#### Applying K-Means Clustering
```python
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Age', 'Spending Score (1-100)']])
```
- Clusters are formed based on Age and Spending Score.

### 5. Visualizing Clusters
```python
plt.scatter(df['Age'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color='black', marker='X')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clusters')
plt.show()
```
- Displays clusters in a 2D scatter plot.
- Black markers indicate cluster centroids.

#### 3D Visualization (Age, Income, Spending Score)
```python
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Cluster'], cmap='rainbow')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlabel('Spending Score')
plt.title('3D Cluster Visualization')
plt.show()
```
Provides a 3D representation of the clusters.

## Key Insights
- **Customers with high spending scores but low income** may be impulse buyers.
- **High-income customers with low spending scores** may be brand-loyal customers.
- **Young customers with high spending scores** may be trend-followers.
- **Segmenting customers helps in targeted marketing strategies.**

## Conclusion
This project successfully clusters customers based on their age, annual income, and spending score. The K-Means algorithm provides meaningful customer segments that businesses can leverage for personalized marketing and decision-making.

## Technologies Used
- **Python** (Data Analysis & Machine Learning)
- **Pandas, NumPy** (Data Handling)
- **Seaborn, Matplotlib** (Visualization)
- **Scikit-Learn** (Machine Learning - K-Means Clustering)

## Future Improvements
- Using **Hierarchical Clustering** for comparison.
- Implementing **DBSCAN** to handle noise and density-based clusters.
- Deploying the model as an interactive dashboard.

---
### 📌 Feel free to contribute or fork this project!

