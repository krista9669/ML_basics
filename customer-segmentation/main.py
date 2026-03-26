# Unsupervised Learning -> Clustering
# The model discovers patterns by itself 

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")
print(data.head())

# features are loaded
features = ["Age", "Annual_Income_(k$)", "Spending_Score"]
X = data[features]

# features are scaled
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# If not scaled:
# Income (large values) dominates
# Age (small values) is ignored

# Scaling ensures:
# Fair comparison
# Meaningful clusters

k = 5   # group customers into 5 segments

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
# fit learns cluster centres and predict assigns cluster labels

data["Cluster"] = clusters

print("\nCluster distribution:")
print(data["Cluster"].value_counts())

print("\nSample clustered data:")
print(data.head())
data.to_csv("clustered_customers.csv", index=False)

print("\nClustering completed. Results saved to clustered_customers.csv")