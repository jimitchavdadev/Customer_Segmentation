# Mall Customers Segmentation Analysis

## Overview
This project performs an exploratory data analysis (EDA) and customer segmentation using the Mall Customers dataset. The primary objective is to identify distinct customer groups based on demographic and behavioral attributes such as age, annual income, and spending score. The analysis leverages data visualization techniques and unsupervised machine learning (K-Means clustering) to uncover patterns and provide actionable insights for targeted marketing strategies.

The dataset used in this project is sourced from Kaggle: **Mall Customers Dataset**.

## Objectives
### Exploratory Data Analysis (EDA):
- Understand the distribution of key variables: Age, Annual Income, and Spending Score.
- Analyze relationships between variables and gender.
- Identify patterns and trends in customer demographics and spending behavior.

### Customer Segmentation:
- Apply K-Means clustering to segment customers based on Age, Annual Income, and Spending Score.
- Determine the optimal number of clusters using the Elbow Method.
- Visualize and interpret the resulting clusters in 2D and 3D spaces.

### Actionable Insights:
- Provide a summary of customer segments to support personalized marketing and business strategies.
- Save the clustered dataset for future use.

## Dataset Description
The **Mall Customers** dataset contains information about customers visiting a mall. It includes the following columns:
- **CustomerID**: Unique identifier for each customer (dropped during analysis as it lacks analytical value).
- **Gender**: Customer's gender (Male/Female).
- **Age**: Customer's age in years.
- **Annual Income (k$)**: Customer's annual income in thousands of dollars.
- **Spending Score (1-100)**: Score assigned by the mall based on customer spending behavior and loyalty (1 = lowest, 100 = highest).

The dataset consists of **200 rows and 5 columns** initially.

## Theoretical Background
### Exploratory Data Analysis (EDA)
EDA is a critical step in data science to summarize the main characteristics of a dataset using statistical and visualization techniques. In this project, EDA helps understand the distribution of variables, detect outliers, and identify relationships between features.

**Tools Used:** Histograms, violin plots, scatter plots, and bar charts to visualize distributions and relationships.

**Key Metrics:** Mean, standard deviation, and counts for numerical and categorical variables.

### K-Means Clustering
K-Means is an unsupervised machine learning algorithm used for partitioning data into **k distinct clusters** based on feature similarity. It minimizes the within-cluster sum of squares (WCSS) by iteratively assigning data points to clusters and updating cluster centroids.

#### Steps:
1. Initialize k centroids randomly (using k-means++ for better initialization).
2. Assign each data point to the nearest centroid.
3. Recalculate centroids as the mean of assigned points.
4. Repeat until convergence or maximum iterations reached.

**Elbow Method**: Used to determine the optimal number of clusters (k) by plotting WCSS against k and identifying the "elbow" point where adding more clusters yields diminishing returns.

**Standardization**: Features are standardized (zero mean, unit variance) to ensure equal weighting in distance calculations.

## Requirements
To run this code, install the following Python libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Code Structure and Explanation
The code is written in Python using a Jupyter Notebook-style format with markdown cells for separation. Below is a detailed breakdown of each section:

### 1. Import Libraries and Set Visualization Styles
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
```

### 2. Load and Inspect the Dataset
```python
df = pd.read_csv('Mall_Customers.csv')
df.head()
```

### 3. Data Preprocessing
```python
df.rename(columns={'Genre':'Gender'}, inplace=True)
df.drop(["CustomerID"], axis=1, inplace=True)
```

### 4. Basic Data Exploration
```python
df.shape
df.describe()
df.dtypes
df.isnull().sum()
```

### 5. Visualizing Distributions
```python
plt.figure(figsize=(18, 6))
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    sns.histplot(df[x], kde=True, bins=20)
```

### 6. Gender Distribution
```python
sns.countplot(y='Gender', data=df)
```

### 7. K-Means Clustering (2D: Age vs. Spending Score)
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X1_scaled)
    wcss.append(kmeans.inertia_)
```

### 8. K-Means Clustering (3D: Age, Income, Spending Score)
```python
X3 = df.loc[:, ["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
X3_scaled = scaler.fit_transform(X3)
kmeans_3d = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
clusters = kmeans_3d.fit_predict(X3_scaled)
```

### 9. Save Results
```python
df.to_csv('Mall_Customers_Clustered.csv', index=False)
```

## Results
- **Total Customers:** 200
- **Optimal Clusters:** 5 (based on Elbow Method and interpretability)

### Key Visualizations:
- Distributions of Age, Income, and Spending Score.
- Gender-based comparisons.
- 2D and 3D cluster visualizations.

**Output:** Clustered dataset saved as `Mall_Customers_Clustered.csv`.

## Usage
1. Download the dataset from Kaggle.
2. Place `Mall_Customers.csv` in the same directory as the script.
3. Run the code in a Python environment with the required libraries installed.

## Future Improvements
- Incorporate additional clustering algorithms (e.g., DBSCAN, Hierarchical Clustering).
- Explore feature engineering (e.g., interaction terms).
- Add silhouette score analysis for cluster validation.
- Build a predictive model based on the clusters.

## License
This project is licensed under the **MIT License**. Feel free to use and modify the code for your purposes.
