# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    silhouette_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# %%
RS = 42

# Output directory
OUTPUT_DIR = Path("Exp03/K-mean-Cluster")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
def majority_cluster_map(cluster_ids, y_true):
    mapping = {}

    for cluster in np.unique(cluster_ids):
        mask = cluster_ids == cluster
        true_labels = y_true[mask]
        majority_label = pd.Series(true_labels).value_counts().idxmax()
        mapping[cluster] = majority_label

    return mapping

# %%
def apply_cluster_map(cluster_ids, mapping):
    return np.array([mapping[c] for c in cluster_ids])

# %%
# 1. Load dataset and use first 8 features
df = pd.read_csv("data/housing_dat.csv")
X = df.iloc[:, :8].values
y_raw = df.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

feature_names = list(df.columns[:8])
df_features = pd.DataFrame(X, columns=feature_names)


print("Dataset shape:", df_features.shape)
print(df_features.head())

# %%
# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features[feature_names])

# %%
# 3. Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=RS, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# %%
# 4. Map clusters to actual labels
cluster_map = majority_cluster_map(clusters, y)
predicted_labels = apply_cluster_map(clusters, cluster_map)

# %%


# %%
# Save evaluation output
with open(OUTPUT_DIR / "evaluation_results.txt", "w") as f:
    f.write(f"Dataset shape: {df_features.shape}\n\n")
    f.write(f"Features used: {feature_names}\n")
    f.write(f"Cluster Mapping: {cluster_map}\n")
    f.write(f"Accuracy after Cluster Mapping: {round(accuracy, 4)}\n")
    f.write(f"Silhouette Score: {round(sil_score, 4)}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nSaved: {OUTPUT_DIR / 'evaluation_results.txt'}")

# %%
# 6. Save predictions
df_features["cluster"] = clusters
df_features["predicted_label"] = predicted_labels
df_features["predicted_label_name"] = label_encoder.inverse_transform(predicted_labels)
df_features["actual_label_name"] = label_encoder.inverse_transform(y)
df_features.to_csv(OUTPUT_DIR / "car_kmeans_predictions.csv", index=False)

print(f"Saved: {OUTPUT_DIR / 'car_kmeans_predictions.csv'}")

# %%
# 7. Plot K-Means clusters using first two features for visualization
plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=clusters,
    cmap="viridis",
    edgecolor="k"
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker="X",
    c="red",
    label="Centroids"
)

plt.title("K-Means Clustering on Car Evaluation Dataset")
plt.xlabel(feature_names[7])
plt.ylabel(feature_names[5])
plt.colorbar(label="Cluster Label")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kmeans_clusters.png", dpi=300)
plt.show()

print(f"Saved: {OUTPUT_DIR / 'kmeans_clusters.png'}")

# %%
# plot 3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 3D Figure
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')

# 3D Scatter Plot
scatter = ax.scatter(
    X_scaled[:, 7],
    X_scaled[:, 5],
    X_scaled[:, 6],
    c=clusters,
    cmap='viridis',
    edgecolor='k',
    label="Centroids"
)

# Labels
ax.set_xlabel(feature_names[7])
ax.set_ylabel(feature_names[5])
ax.set_zlabel(feature_names[6])

ax.set_title('3D K-Means Clustering after PCA')

# Colorbar
plt.colorbar(scatter, label='Cluster Label')
plt.legend()
plt.savefig('Figure/PCA_KMeans_3D.pdf')
plt.show()

# %%



