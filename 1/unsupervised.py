import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1) IMPORTING THE DATA ------------------------------------------------------------------
data = {
    'Customer': ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Helen', 'Ian', 'Jack',
                 'Kate', 'Leo', 'Mona', 'Nina', 'Oscar', 'Paul', 'Quinn', 'Rose', 'Steve', 'Tom'],
    'Monthly Visits': [1, 2, 3, 3, 4, 5, 9, 13, 11, 12,
                       14, 28, 21, 26, 20, 24, 28, 29, 35, 37],
    'Annual Spending ($)': [500, 700, 1200, 1500, 1850, 1250, 3000, 3400, 4000, 4500,
                            5200, 6000, 6500, 7000, 8200, 7700, 7900, 8300, 9500, 8700]
}

df = pd.DataFrame(data)  # Convert dictionary to pandas DataFrame

# 2) TRAINING DATA -----------------------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)  # Now with 4 clusters


# 3) PREDICTING DATA ---------------------------------------------------------------------
df['Cluster'] = kmeans.fit_predict(df[['Monthly Visits', 'Annual Spending ($)']])

# 4) PLOTTING RESULTS --------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# Scatter plot for centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroids')

# Plotting Details
plt.xlabel('Monthly Visits')
plt.ylabel('Annual Spending ($)')
plt.title('Customer Segmentation Using K-Means')

plt.legend()
plt.show()

# Display DataFrame with clusters
print(df)
