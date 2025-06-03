# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', 10)  # or however many rows you want


# %%
DecisionTreeDF = pd.read_csv("Data for Modeling/AftDTsMPBROCKMETAL_KGDf.csv")
df_cluster_PCA_gp = pd.read_csv("Data for Modeling/df_cluster_PCA_gp.csv")

# %%
del DecisionTreeDF[" "]
del DecisionTreeDF["Unnamed: 0"]
del df_cluster_PCA_gp ["Unnamed: 0"]


# %%
DecisionTreeDF


# %%
df_cluster_PCA_gp

# %%
df_cluster_PCA_gp_Genre = pd.merge(DecisionTreeDF["track_genre"],df_cluster_PCA_gp, right_index=True, left_index=True )
df_cluster_PCA_gp_Genre = pd.merge(DecisionTreeDF["track_id"],df_cluster_PCA_gp_Genre, right_index=True, left_index=True )
df_cluster_PCA_gp_Genre.sort_values(by="cluster_x")

# %%
def countingOcorencies(df, columnname, clusterIDs):
    # Count how many of each genre exist per cluster
    grouped = df.groupby([columnname, "track_genre"]).size().unstack(fill_value=0)

    # Make sure all expected clusters are present
    grouped = grouped.reindex(clusterIDs, fill_value=0)

    # Rename and reorder columns for clarity
    grouped = grouped.rename(columns={
        "mpb": "QntMPB",
        "rock": "QntROCK"
    })

    # Add a column for metal (any genre that's not mpb or rock)
    if "QntMPB" not in grouped.columns:
        grouped["QntMPB"] = 0
    if "QntROCK" not in grouped.columns:
        grouped["QntROCK"] = 0

    # Infer "metal" as all remaining genres
    grouped["QntMETAL"] = (
        df.groupby(columnname)
          .size()
          .reindex(clusterIDs, fill_value=0)
          - grouped["QntMPB"]
          - grouped["QntROCK"]
    )

    # Reset index so Cluster becomes a column
    grouped = grouped.reset_index().rename(columns={columnname: "Cluster"})
    grouped = grouped.drop_duplicates()

    # Count total IDs considered
    ids_count = len(df)

    return grouped, ids_count

d, i = countingOcorencies(df_cluster_PCA_gp_Genre, "cluster_x", list(df_cluster_PCA_gp_Genre["cluster_x"]))
d1, i2 = countingOcorencies(DecisionTreeDF, "node_id", (DecisionTreeDF["node_id"].sort_values()).to_list())
d_sorted = d.sort_values("Cluster")
d1_sorted = d1.sort_values("Cluster")

# %%

# Clean any sets (if applicable)
for col in ["QntMPB", "QntROCK", "QntMETAL"]:
    d_sorted[col] = d_sorted[col].apply(lambda x: list(x)[0] if isinstance(x, set) else x)

# Prepare for heatmap
heatmap_data = d_sorted.set_index("Cluster")[["QntMPB", "QntROCK", "QntMETAL"]]

# Plot heatmap
plt.figure(figsize=(8, 20))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt="d")

plt.title("Heatmap of Song Counts per Genre per Cluster")
plt.xlabel("Genre")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()


# %%



for col in ["Cluster", "QntMPB", "QntROCK", "QntMETAL"]:
    d1[col] = d1[col].apply(lambda x: list(x)[0] if isinstance(x, set) else x)

heatmap_data = d1.set_index("Cluster")[["QntMPB", "QntROCK", "QntMETAL"]]

plt.figure(figsize=(8, 20))
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt="d")

plt.title("Heatmap of Song Counts per Genre per Cluster")
plt.xlabel("Genre")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()


# %%



