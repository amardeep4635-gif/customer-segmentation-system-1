import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- STEP 1: DATA COLLECTION  ---
try:
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    # Creating dummy data based on document features [cite: 34, 36, 37]
    data = {
        'CustomerID': range(1, 201),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 130, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    }
    df = pd.DataFrame(data)

# --- STEP 2: DATA PREPROCESSING & EDA [cite: 44, 48] ---
# Selecting Annual Income and Spending Score for visualization [cite: 36, 37, 51]
X = df.iloc[:, [2, 3]].values 

# FIGURE 1: Feature Distributions [cite: 49]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['Annual Income (k$)'], kde=True, ax=ax[0], color='skyblue')
ax[0].set_title('Figure 1a: Annual Income Distribution')
sns.histplot(df['Spending Score (1-100)'], kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Figure 1b: Spending Score Distribution')
plt.show()

# FIGURE 2: Correlation Heatmap [cite: 50]
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Figure 2: Variable Correlation Heatmap')
plt.show()

# --- STEP 3: FEATURE SCALING [cite: 47] ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- STEP 4: K-MEANS CLUSTERING [cite: 53] ---
# FIGURE 3: Elbow Method to find K 
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Figure 3: Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.show()

# Applying K-Means (Optimal K=5) [cite: 56, 88]
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# FIGURE 4: K-Means Cluster Visualization [cite: 85]
plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']

for i in range(5):
    plt.scatter(X_scaled[y_kmeans == i, 0], X_scaled[y_kmeans == i, 1], 
                s=50, c=colors[i], label=labels[i])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='yellow', marker='*', label='Centroids', edgecolors='black')
plt.title('Figure 4: K-Means Customer Segments')
plt.xlabel('Scaled Annual Income')
plt.ylabel('Scaled Spending Score')
plt.legend()
plt.show()

# --- STEP 5: DBSCAN CLUSTERING [cite: 70] ---
# Detects outliers/noise (labeled as -1) [cite: 71, 76]
dbscan = DBSCAN(eps=0.3, min_samples=5) # Parameters [cite: 72]
y_dbscan = dbscan.fit_predict(X_scaled)

# FIGURE 5: DBSCAN Outlier Detection [cite: 93, 94]
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='plasma', s=50)
plt.title('Figure 5: DBSCAN Clustering (Detecting Outliers)')
plt.xlabel('Scaled Annual Income')
plt.ylabel('Scaled Spending Score')
plt.show()

# --- STEP 6: MODEL EVALUATION [cite: 81] ---
# Calculating Silhouette Score [cite: 63, 83]
k_score = silhouette_score(X_scaled, y_kmeans)
print(f"Final Results:")
print(f"1. K-Means Silhouette Score: {k_score:.4f}")
print(f"2. Clusters identified: {np.unique(y_kmeans)}")
# --- STEP 7: DEEPER CLUSTER ANALYSIS ---
# Adding cluster labels back to the original dataframe for analysis
df['Cluster'] = y_kmeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# --- STEP 1: DATA COLLECTION [cite: 42] ---
try:
    # Attempting to load the recommended Mall Customer dataset [cite: 40]
    df = pd.read_csv('Mall_Customers.csv')
except FileNotFoundError:
    # Fallback to dummy data using project features: Age, Income, Spending [cite: 34, 36, 37]
    np.random.seed(42)
    data = {
        'CustomerID': range(1, 201),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 130, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    }
    df = pd.DataFrame(data)

# --- STEP 2: DATA PREPROCESSING & EDA [cite: 44, 48] ---
# Selecting features for clustering [cite: 51]
X_raw = df[['Annual Income (k$)', 'Spending Score (1-100)']].values 

# FIGURE 1: Feature Distributions [cite: 49]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(df['Annual Income (k$)'], kde=True, ax=ax[0], color='skyblue')
ax[0].set_title('Figure 1a: Annual Income Distribution')
sns.histplot(df['Spending Score (1-100)'], kde=True, ax=ax[1], color='salmon')
ax[1].set_title('Figure 1b: Spending Score Distribution')
plt.tight_layout()
plt.show()

# FIGURE 2: Correlation Heatmap [cite: 50]
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Figure 2: Variable Correlation Heatmap')
plt.show()

# Feature Scaling (StandardScaler) [cite: 47]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- STEP 3: K-MEANS CLUSTERING [cite: 53] ---
# FIGURE 3: Elbow Method to find optimal K [cite: 62]
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Figure 3: Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.show()

# Applying K-Means with K=5 [cite: 56, 88]
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# FIGURE 4: K-Means Visualization [cite: 85]
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=df['Cluster'], palette='viridis', s=70)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='red', marker='X', label='Centroids')
plt.title('Figure 4: K-Means Customer Segments')
plt.legend(title='Cluster')
plt.show()

# --- STEP 4: DBSCAN CLUSTERING [cite: 70] ---
# Density-based clustering to detect noise/outliers [cite: 71, 76]
dbscan = DBSCAN(eps=0.3, min_samples=5) # Parameters: eps and min_samples [cite: 72]
y_dbscan = dbscan.fit_predict(X_scaled)

# FIGURE 5: DBSCAN Outlier Detection 
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='plasma', s=50)
plt.title('Figure 5: DBSCAN Clustering (Outliers = -1)')
plt.show()

# --- STEP 5: BUSINESS INSIGHTS & EVALUATION [cite: 23, 81] ---
# FIGURE 6: Age Distribution per Cluster
plt.figure(figsize=(7, 4))
sns.boxplot(x='Cluster', y='Age', data=df, palette='Set2')
plt.title('Figure 6: Age Distribution per Segment')
plt.show()

# FIGURE 7: Feature Relationships [cite: 111]
sns.pairplot(df.drop('CustomerID', axis=1), hue='Cluster', palette='bright')
plt.suptitle('Figure 7: Multidimensional Relationships', y=1.02)
plt.show()

# FIGURE 8: Cluster Mean Profiles 
plt.figure(figsize=(8, 5))
df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().plot(kind='bar')
plt.title('Figure 8: Average Income & Spending per Cluster')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.show()

# FIGURE 9: Silhouette Score Evaluation [cite: 63, 83]
k_score = silhouette_score(X_scaled, df['Cluster'])
plt.figure(figsize=(7, 5))
s_values = silhouette_samples(X_scaled, df['Cluster'])
y_low = 10
for i in range(5):
    ith_values = s_values[df['Cluster'] == i]
    ith_values.sort()
    y_up = y_low + len(ith_values)
    plt.fill_betweenx(np.arange(y_low, y_up), 0, ith_values, alpha=0.7)
    y_low = y_up + 10
plt.title(f'Figure 9: Silhouette Analysis (Avg Score: {k_score:.2f})')
plt.show()

# FIGURE 10: Final Algorithm Comparison [cite: 21]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis')
ax1.set_title('Figure 10a: K-Means Groups')
ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='plasma')
ax2.set_title('Figure 10b: DBSCAN Outliers')
plt.tight_layout()
plt.show()

print(f"K-Means Silhouette Score: {k_score:.4f}")
