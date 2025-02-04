import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'players_cleaned.csv'
players_data = pd.read_csv(file_path)

# Select numeric columns
numeric_data = players_data.select_dtypes(include=['float64', 'int64'])

# Calculate variances and select top features
variances = numeric_data.var().sort_values(ascending=False)
high_variance_features = variances.head(15).index  # Top 15 most variable features
selected_data = numeric_data[high_variance_features]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)

# Reduce dimensionality using PCA to improve clustering
pca = PCA(n_components=10)  # Focus on 10 most important dimensions
reduced_data = pca.fit_transform(scaled_data)

# Apply OPTICS clustering with fine-tuned parameters
optics_model = OPTICS(min_samples=2, xi=0.005, min_cluster_size=0.005, metric='euclidean', cluster_method='xi')
optics_model.fit(reduced_data)

# Handle unclustered points (-1) by assigning them to the nearest cluster
cluster_labels = optics_model.labels_
if -1 in cluster_labels:
    cluster_centers = [
        reduced_data[cluster_labels == cluster].mean(axis=0)
        for cluster in np.unique(cluster_labels) if cluster != -1
    ]
    cluster_centers = np.array(cluster_centers)

    for i, label in enumerate(cluster_labels):
        if label == -1:  # Assign unclustered points
            distances = np.linalg.norm(reduced_data[i] - cluster_centers, axis=1)
            cluster_labels[i] = np.argmin(distances)

# Map cluster labels to meaningful names
unique_clusters = sorted(set(cluster_labels))
label_mapping = {}
for i, cluster in enumerate(unique_clusters):
    if i == 0:
        label_mapping[cluster] = 'Low'
    elif i == 1:
        label_mapping[cluster] = 'Medium'
    elif i == 2:
        label_mapping[cluster] = 'High'
    elif i == 3:
        label_mapping[cluster] = 'Very High'
    else:
        label_mapping[cluster] = 'Elite'

# Assign cluster names to players
players_data['predicted_performance'] = [label_mapping[label] for label in cluster_labels]

# Define performance categories based on overall_rating, using the same names
def calculate_performance(overall_rating):
    if 50 <= overall_rating < 50:
        return 'Low'
    elif 50 <= overall_rating < 65:
        return 'Medium'
    elif 65 <= overall_rating < 85:
        return 'High'
    elif 85 <= overall_rating < 90:
        return 'Very High'
    elif 90 <= overall_rating <= 94:
        return 'Elite'

players_data['performance'] = players_data['overall_rating'].apply(calculate_performance)

# Create a random performance column
performance_levels = ['Low', 'Medium', 'High', 'Very High', 'Elite']
np.random.seed(42)  # For reproducibility
players_data['random_performance'] = np.random.choice(performance_levels, size=len(players_data))

# Calculate accuracy of predicted_performance vs performance
correct_matches = (players_data['predicted_performance'] == players_data['performance']).sum()
accuracy = correct_matches / len(players_data) * 100

# Calculate accuracy of random_performance vs performance
random_matches = (players_data['random_performance'] == players_data['performance']).sum()
random_accuracy = random_matches / len(players_data) * 100

# Create a scatter plot with a legend for cluster labels
plt.figure(figsize=(10, 6))
colors = ['yellow', 'green', 'cyan', 'purple', 'red']  # Assign specific colors to clusters
for cluster in unique_clusters:
    cluster_data = reduced_data[cluster_labels == cluster]
    plt.scatter(
        cluster_data[:, 0], cluster_data[:, 1],
        label=f"Cluster {cluster}: {label_mapping[cluster]}",
        s=50, alpha=0.7, c=colors[cluster]
    )
plt.title('OPTICS Clustering of Players (Enhanced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid()
plt.show()

# Save the results to a CSV file with only the required columns
output_path = 'players_optics.csv'
players_data[['name', 'performance', 'predicted_performance', 'random_performance']].to_csv(output_path, index=False)
silhouette_vals = silhouette_samples(reduced_data, cluster_labels)
average_silhouette = silhouette_score(reduced_data, cluster_labels)
# Print results
print(f"Accuracy of predicted_performance vs performance: {accuracy:.2f}%")
print(f"Accuracy of random_performance vs performance: {random_accuracy:.2f}%")
print(f"Clusters created using OPTICS and saved as '{output_path}'.")

print(f"Average Silhouette Score: {average_silhouette:.2f}")
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'players_cleaned.csv'
players_data = pd.read_csv(file_path)

# Define performance categories based on overall_rating
def calculate_performance(overall_rating):
    if 50 <= overall_rating < 60:
        return 'Low'
    elif 60 <= overall_rating < 70:
        return 'Medium'
    elif 70 <= overall_rating < 80:
        return 'High'
    elif 80 <= overall_rating < 90:
        return 'Very High'
    elif 90 <= overall_rating <= 94:
        return 'Elite'

# Add performance column
players_data['performance'] = players_data['overall_rating'].apply(calculate_performance)

# Drop rows with missing values
players_data.dropna(inplace=True)

# Select only numeric columns for correlation calculation
numeric_data = players_data.select_dtypes(include=['float64', 'int64'])

# Calculate correlations with overall_rating
correlation = numeric_data.corrwith(players_data['overall_rating']).sort_values(ascending=False)
print("Correlation with overall_rating:")
print(correlation)

# Remove highly correlated features
high_corr_features = correlation[correlation > 0.95].index
X = players_data.drop(columns=['performance', 'overall_rating', *high_corr_features])  # Drop correlated features
X = X.select_dtypes(include=['float64', 'int64'])  # Ensure only numeric columns remain
y = players_data['performance']  # Target

# Split dataset into train2 (80%), val2 (10%), and test2 (10%)
X_train2, X_temp, y_train2, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val2, X_test2, y_val2, y_test2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=5)  # Reduce to 5 principal components
X_train2 = pca.fit_transform(X_train2)
X_val2 = pca.transform(X_val2)
X_test2 = pca.transform(X_test2)

# Train a simplified Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)  # Reduced complexity
rf_model.fit(X_train2, y_train2)

# Predict on the validation set
y_val2_pred = rf_model.predict(X_val2)

# Evaluate the model on validation set
val_accuracy = accuracy_score(y_val2, y_val2_pred)
print(f"Validation Accuracy: {val_accuracy:.2f}")
print("\nClassification Report on Validation Set:")
print(classification_report(y_val2, y_val2_pred))

# Predict on the test set
y_test2_pred = rf_model.predict(X_test2)

# Save predictions for test2 (only true and predicted performance)
test2_predictions = pd.DataFrame({
    'true_performance': y_test2,
    'predicted_performance': y_test2_pred
})
test2_predictions.to_csv('test2_predictions_random_forest.csv', index=False)

print(f"Test predictions saved to: test2_predictions_random_forest.csv")
