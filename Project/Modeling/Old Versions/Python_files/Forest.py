# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.tree import _tree, plot_tree

from matplotlib.gridspec import GridSpec
import joblib
from models_and_utils.tree_utils import *


# %%
df = pd.read_csv("Data for Modeling/MPBROCKMETAL_KGsubset.csv")

# %%


# === 1. Prepare Data ===
X = df.iloc[:, 6:20]
y = df["track_genre"]
feature_names = X.columns

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=feature_names)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = label_encoder.classes_


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.25, random_state=42
)

# === 2. Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, criterion="entropy")
clf.fit(X_train, y_train)

# === 3. Find Best Tree by Accuracy ===
best_accuracy = 0
best_tree_index = 0

for i, tree in enumerate(clf.estimators_):
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_tree_index = i

print(f"Best tree index (accuracy): {best_tree_index}")
print(f"Best tree accuracy: {best_accuracy:.4f}")

best_tree = clf.estimators_[best_tree_index]

rules = extract_rules(
    tree=best_tree,
    feature_names=X_train.columns,
    X_data=X_train,
    index_data=X_train.index
)
rules_df = pd.DataFrame(rules)

print("\nSample of extracted rules:")
print(rules_df.head())

# === 5. Evaluation: Forest vs Best Tree ===
forest_pred = clf.predict(X_test)
tree_pred = best_tree.predict(X_test)

forest_acc = accuracy_score(y_test, forest_pred)
tree_acc = accuracy_score(y_test, tree_pred)

forest_probs = clf.predict_proba(X_test)
tree_probs = best_tree.predict_proba(X_test)

forest_logloss = log_loss(y_test, forest_probs, labels=range(len(class_labels)))
tree_logloss = log_loss(y_test, tree_probs, labels=range(len(class_labels)))

print(f"\nRandom Forest accuracy: {forest_acc:.4f}")
print(f"Best Tree accuracy:     {tree_acc:.4f}")
print(f"Random Forest log loss: {forest_logloss:.4f}")
print(f"Best Tree log loss:     {tree_logloss:.4f}")





# %%
y_decoded = label_encoder.inverse_transform(y_encoded)

cm = confusion_matrix(y_test, tree_pred)
class_labels = label_encoder.classes_

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=True,
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
plt.show()

# %%

original_labels = label_encoder.inverse_transform(y_encoded)
class_names = label_encoder.classes_  # Returns array like: ['black metal' 'mpb' 'rock']


plt.figure(figsize=(80, 40))  
plot_tree(best_tree,
          feature_names=feature_names,
          class_names=class_names,  
          filled=True,
          node_ids=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree - Matplotlib Visualization")
plt.show()


# %%
importances = pd.Series(best_tree.feature_importances_, index=X.columns)
importances

# %%


specific_rules = pd.concat([rules_df.loc[rules_df["node"] == 1], rules_df.loc[rules_df["node"] == 86],rules_df.loc[rules_df["node"] == 87],rules_df.loc[rules_df["node"] == 136]])

print(specific_rules)


# %%


def plot_specific_thresholds(df, specific_rules, index_col="index", scatter_color='blue'):
    n_rules = len(specific_rules)
    
    # Define grid size for 4 plots (2 rows x 2 cols)
    n_cols = 2
    n_rows = int(np.ceil(n_rules / n_cols))
    
    fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    for plot_idx, (_, rule) in enumerate(specific_rules.iterrows()):
        feature = rule['feature']
        threshold = rule['threshold']
        data_indices = rule['data_indices']

        # Filter rows using the index_col and the indices from specific_rules
        mask = df[index_col].isin(data_indices)
        subset = df.loc[mask]
        
        # Reorder subset to maintain order in data_indices
        subset = subset.set_index(index_col).loc[data_indices].reset_index()
        
        x = subset[feature].values
        
        if len(x) == 0:
            print(f"Warning: No data found for feature '{feature}' at node {rule['node']} with given indices.")
            continue
        
        ax = fig.add_subplot(gs[plot_idx])
        
        x_min, x_max = x.min() - 0.1 * abs(x.min()), x.max() + 0.1 * abs(x.max())
        ax.set_xlim(x_min, x_max)
        
        # Plot all points in the subset as blue dots on x-axis (y=0)
        ax.scatter(x, np.zeros_like(x), color=scatter_color, edgecolor='k', s=30, alpha=0.7)
        
        # Plot vertical threshold line
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel(feature)
        ax.set_yticks([])
        ax.set_title(f'Node {rule["node"]} - {feature} ≤ {threshold:.3f}')
    
    plt.tight_layout()
    plt.show()



plot_specific_thresholds(X.reset_index(), specific_rules)


