Mall Customers Segmentation Analysis
Overview
This project performs an exploratory data analysis (EDA) and customer segmentation using the Mall Customers dataset. The primary objective is to identify distinct customer groups based on demographic and behavioral attributes such as age, annual income, and spending score. The analysis leverages data visualization techniques and unsupervised machine learning (K-Means clustering) to uncover patterns and provide actionable insights for targeted marketing strategies.
The dataset used in this project is sourced from Kaggle: Mall Customers Dataset.
Objectives
Exploratory Data Analysis (EDA):
Understand the distribution of key variables: Age, Annual Income, and Spending Score.

Analyze relationships between variables and gender.

Identify patterns and trends in customer demographics and spending behavior.

Customer Segmentation:
Apply K-Means clustering to segment customers based on Age, Annual Income, and Spending Score.

Determine the optimal number of clusters using the Elbow Method.

Visualize and interpret the resulting clusters in 2D and 3D spaces.

Actionable Insights:
Provide a summary of customer segments to support personalized marketing and business strategies.

Save the clustered dataset for future use.

Dataset Description
The Mall Customers dataset contains information about customers visiting a mall. It includes the following columns:
CustomerID: Unique identifier for each customer (dropped during analysis as it lacks analytical value).

Gender: Customer's gender (Male/Female).

Age: Customer's age in years.

Annual Income (k$): Customer's annual income in thousands of dollars.

Spending Score (1-100): Score assigned by the mall based on customer spending behavior and loyalty (1 = lowest, 100 = highest).

The dataset consists of 200 rows and 5 columns initially.
Theoretical Background
Exploratory Data Analysis (EDA)
EDA is a critical step in data science to summarize the main characteristics of a dataset using statistical and visualization techniques. In this project, EDA helps understand the distribution of variables, detect outliers, and identify relationships between features.
Tools Used: Histograms, violin plots, scatter plots, and bar charts to visualize distributions and relationships.

Key Metrics: Mean, standard deviation, and counts for numerical and categorical variables.

K-Means Clustering
K-Means is an unsupervised machine learning algorithm used for partitioning data into k distinct clusters based on feature similarity. It minimizes the within-cluster sum of squares (WCSS) by iteratively assigning data points to clusters and updating cluster centroids.
Steps:
Initialize k centroids randomly (using k-means++ for better initialization).

Assign each data point to the nearest centroid.

Recalculate centroids as the mean of assigned points.

Repeat until convergence or maximum iterations reached.

Elbow Method: Used to determine the optimal number of clusters (k) by plotting WCSS against k and identifying the "elbow" point where adding more clusters yields diminishing returns.

Standardization: Features are standardized (zero mean, unit variance) to ensure equal weighting in distance calculations.

Requirements
To run this code, install the following Python libraries:
bash

pip install numpy pandas matplotlib seaborn scikit-learn

Code Structure and Explanation
The code is written in Python using a Jupyter Notebook-style format with markdown cells (# %%) for separation. Below is a detailed breakdown of each section:
1. Import Libraries and Set Visualization Styles
python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

Loads essential libraries for data manipulation (numpy, pandas), visualization (matplotlib, seaborn), and clustering (sklearn).

Configures plot aesthetics for better readability and clarity.

2. Load and Inspect the Dataset
python

df = pd.read_csv('Mall_Customers.csv')
df.head()

Reads the dataset from a CSV file and displays the first 5 rows to verify the structure.

3. Data Preprocessing
python

df.rename(columns={'Genre':'Gender'}, inplace=True)
df.drop(["CustomerID"], axis=1, inplace=True)

Fixes a typo in the column name (Genre → Gender).

Drops CustomerID as it does not contribute to the analysis.

4. Basic Data Exploration
python

df.shape  # (200, 4) after dropping CustomerID
df.describe()  # Statistical summary
df.dtypes  # Data types
df.isnull().sum()  # Check for missing values (none found)

Checks dataset dimensions, statistical summaries, data types, and missing values.

5. Visualizing Distributions
python

plt.figure(figsize=(18, 6))
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    sns.histplot(df[x], kde=True, bins=20)

Creates histograms with kernel density estimates (KDE) for numerical variables to understand their distributions.

6. Gender Distribution
python

sns.countplot(y='Gender', data=df)

Visualizes the count of male and female customers with annotated labels.

7. Violin Plots by Gender
python

for cols in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    sns.violinplot(x=cols, y='Gender', data=df, palette='Set2')

Compares the distribution of numerical variables across genders using violin plots.

8. Age Group Analysis
python

df['Age_Group'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 100], labels=["18-25", "26-35", "36-45", "46-55", "55+"])
sns.barplot(x=df['Age_Group'].value_counts().index, y=df['Age_Group'].value_counts().values)

Categorizes customers into age groups and visualizes the counts.

9. Income vs. Spending Score Relationship
python

sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df, hue="Gender", size="Age")

Explores the relationship between income and spending score, with gender and age as additional dimensions.

10. Spending Score and Income Group Analysis
python

df['Spending_Group'] = pd.cut(df['Spending Score (1-100)'], bins=[0, 20, 40, 60, 80, 100], labels=["1-20", "21-40", "41-60", "61-80", "81-100"])
df['Income_Group'] = pd.cut(df['Annual Income (k$)'], bins=[0, 30, 60, 90, 120, 150], labels=["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"])

Segments spending scores and annual incomes into groups and visualizes their distributions.

11. K-Means Clustering (2D: Age vs. Spending Score)
python

X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X1_scaled)
    wcss.append(kmeans.inertia_)

Extracts features, standardizes them, and applies the Elbow Method to find the optimal k.

Clusters with k=4 and k=5 are tested and visualized with centroids and labels.

12. K-Means Clustering (3D: Age, Income, Spending Score)
python

X3 = df.loc[:, ["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
X3_scaled = scaler_3d.fit_transform(X3)
kmeans_3d = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
clusters = kmeans_3d.fit_predict(X3_scaled)

Extends clustering to 3D with k=5 and visualizes results using a 3D scatter plot.

13. Cluster Analysis and Summary
python

cluster_summary = df.groupby('label').agg({
    'Age': ['mean', 'std'],
    'Annual Income (k$)': ['mean', 'std'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Gender': lambda x: x.mode()[0]
})

Summarizes cluster characteristics (mean, standard deviation, and mode gender).

Visualizes distributions using box plots.

14. Save Results
python

df.to_csv('Mall_Customers_Clustered.csv', index=False)

Saves the clustered dataset for future use.

Results
Total Customers: 200

Optimal Clusters: 5 (based on Elbow Method and interpretability)

Key Visualizations:
Distributions of Age, Income, and Spending Score.

Gender-based comparisons.

2D and 3D cluster visualizations.

Output: Clustered dataset saved as Mall_Customers_Clustered.csv.

Usage
Download the dataset from Kaggle.

Place Mall_Customers.csv in the same directory as the script.

Run the code in a Python environment with the required libraries installed.

Future Improvements
Incorporate additional clustering algorithms (e.g., DBSCAN, Hierarchical Clustering).

Explore feature engineering (e.g., interaction terms).

Add silhouette score analysis for cluster validation.

Build a predictive model based on the clusters.

License
This project is licensed under the MIT License. Feel free to use and modify the code for your purposes.

