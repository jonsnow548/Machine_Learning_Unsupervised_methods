import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'players_cleaned.csv'  # Update the file path if necessary
players_data = pd.read_csv(file_path)

# Select relevant columns for clustering
numerical_data = players_data[['offensive_power', 'defensive_coverage']]

# Standardize the data
scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data)
silhouette_scores = []
k_values = range(2, 11)  # Testing different numbers of clusters (2 to 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(numerical_data_scaled)
    silhouette_avg = silhouette_score(numerical_data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid()
plt.show()

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
players_data['kmeans_cluster'] = kmeans.fit_predict(numerical_data_scaled)

# Evaluate clustering with Silhouette Score
silhouette_avg = silhouette_score(numerical_data_scaled, players_data['kmeans_cluster'])
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Map clusters to roles (manually assigned for clarity)
role_mapping = {
    2: 'Forward',
    3: 'Defender',
    0: 'Midfielder',
    1: 'Goalkeeper'
}
players_data['predicted_role'] = players_data['kmeans_cluster'].map(role_mapping)

# Generate a random role column for comparison
roles = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
players_data['random_role'] = np.random.choice(roles, len(players_data))

# Calculate equivalence percentages
predicted_vs_actual = (players_data['predicted_role'] == players_data['role']).mean() * 100
actual_vs_random = (players_data['role'] == players_data['random_role']).mean() * 100

print(f"Percentage of Predicted Roles matching Actual Roles: {predicted_vs_actual:.2f}%")
print(f"Percentage of Actual Roles matching Random Roles: {actual_vs_random:.2f}%")

# Save the results
final_output = players_data[['name', 'predicted_role', 'role', 'random_role']]
output_path = 'k_means.csv'
final_output.to_csv(output_path, index=False)

# Plot the clusters with player names
plt.figure(figsize=(12, 8))
colors = ['orange', 'blue', 'green', 'red']
for cluster in range(4):
    cluster_data = numerical_data_scaled[players_data['kmeans_cluster'] == cluster]
    cluster_names = players_data.loc[players_data['kmeans_cluster'] == cluster, 'name'].sample(5, random_state=42)

    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=role_mapping.get(cluster, f'Cluster {cluster}'), alpha=0.6, c=colors[cluster])

    # Annotate with player names
    for i, name in enumerate(cluster_names):
        player_idx = players_data[(players_data['name'] == name) & (players_data['kmeans_cluster'] == cluster)].index[0]
        plt.annotate(name, (numerical_data_scaled[player_idx, 0], numerical_data_scaled[player_idx, 1]), fontsize=9, alpha=0.7)

plt.title('K-Means Clustering of Players')
plt.xlabel('Standardized Feature 1 (Offensive Power)')
plt.ylabel('Standardized Feature 2 (Defensive Coverage)')
plt.legend()
plt.grid()
plt.show()

print(f"File saved to: {output_path}")
