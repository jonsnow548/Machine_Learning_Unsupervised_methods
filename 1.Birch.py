import numpy as np
import pandas as pd
from sklearn.cluster import Birch
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

# Apply BIRCH clustering with improved parameters
birch = Birch(n_clusters=4, threshold=0.4, branching_factor=50)  # Optimized parameters
players_data['birch_cluster'] = birch.fit_predict(numerical_data_scaled)

# Evaluate clustering with Silhouette Score
silhouette_avg = silhouette_score(numerical_data_scaled, players_data['birch_cluster'])
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Map clusters to roles (manually assigned for clarity)
role_mapping = {
    0: 'Forward',
    1: 'Defender',
    3: 'Midfielder',
    2: 'Goalkeeper'
}
players_data['predicted_role'] = players_data['birch_cluster'].map(role_mapping)

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
output_path = 'players_with_optics_performance_filtered.csv'
final_output.to_csv(output_path, index=False)

# Plot the clusters with player names
plt.figure(figsize=(12, 8))
colors = ['orange', 'blue', 'green', 'red']
for cluster in range(4):
    cluster_data = numerical_data_scaled[players_data['birch_cluster'] == cluster]
    cluster_names = players_data.loc[players_data['birch_cluster'] == cluster, 'name'].sample(5, random_state=42)

    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=role_mapping.get(cluster, f'Cluster {cluster}'), alpha=0.6, c=colors[cluster])

    # Annotate with player names
    for i, name in enumerate(cluster_names):
        player_idx = players_data[(players_data['name'] == name) & (players_data['birch_cluster'] == cluster)].index[0]
        plt.annotate(name, (numerical_data_scaled[player_idx, 0], numerical_data_scaled[player_idx, 1]), fontsize=9, alpha=0.7)

plt.title('Optimized BIRCH Clustering of Players')
plt.xlabel('Standardized Feature 1 (Offensive Power)')
plt.ylabel('Standardized Feature 2 (Defensive Coverage)')
plt.legend()
plt.grid()
plt.show()

print(f"File saved to: {output_path}")
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
import random
import matplotlib.pyplot as plt

# Reading the CSV files
train = pd.read_csv('train.csv')
test_input = pd.read_csv('test_input.csv')  # Without the 'role' column
test_full = pd.read_csv('test_full.csv')  # With the 'role' column for evaluation
val = pd.read_csv('val.csv')

# Selecting numerical attributes and the target label 'role'
numerical_columns = train.select_dtypes(include=['float64', 'int64']).columns

X_train = train[numerical_columns]  # Numerical attributes for training
y_train = train['role']  # Target label for training

X_test = test_input[numerical_columns]  # Numerical attributes for testing
y_test = test_full['role']  # Target label for testing

X_val = val[numerical_columns]  # Numerical attributes for validation
y_val = val['role']  # Target label for validation

# Training a RandomForest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluating the model on the test set
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Displaying the results
print(f"Test set accuracy: {accuracy:.4f}")
print("\nClassification report:")
print(report)

# Creating a file with predictions
test_full['predicted_role'] = y_pred  # Adding the predicted roles
predictions_output_path = 'test_predictions.csv'
test_full[['name', 'role', 'predicted_role']].to_csv(predictions_output_path, index=False)

print(f"The predictions file has been saved to: {predictions_output_path}")
