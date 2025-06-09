# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
from models_and_utils.tree_utils import *
import pandas as pd
import joblib


# %%
df = pd.read_csv("Data for Modeling/MPBROCKMETAL_KGsubset.csv")
scaler = joblib.load("scaler_encoder/scaler.pkl")
label_encoder = joblib.load("scaler_encoder/label_encoder.pkl") 
best_tree = joblib.load("models_and_utils/best_tree.pkl")


# %%
del df["Unnamed: 0"]
del df["Unnamed: 0.1"]

# %%

def pDecisionTree(Xdata, ydata):
    feature_names = Xdata.columns

    # Scale new data
    X_new_scaled = scaler.transform(Xdata)
    X_new = pd.DataFrame(X_new_scaled, columns=feature_names)
    y_true_encoded = label_encoder.transform(ydata)
    class_labels = label_encoder.classes_

    # Predict using the already-trained best_tree
    tree_pred = best_tree.predict(X_new)


    return X_new,tree_pred, y_true_encoded, class_labels  # Return both to allow accuracy calculation

def getAccuracy(y_pred, y_true_encoded, class_labels):
    acc = accuracy_score(y_true_encoded, y_pred)
    print(f"Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_true_encoded, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(xticks_rotation=45, cmap='Blues')
    

def add_encoded_labels_to_df(y_true_encoded, df):
    df_y_encoded = pd.DataFrame({"track_genre_encoded": y_true_encoded})
    df_combined = pd.merge(df, df_y_encoded, left_index=True, right_index=True)
    return df_combined

# %%
Xs = df.iloc[:, 4:18]
y = df["track_genre"]

t, x, y,z= pDecisionTree(Xs,y)
df_encoded_genr = add_encoded_labels_to_df(y, df)
getAccuracy(x,y,z)

# %%
df_with_leaf_nodes =  get_samples_leaf_nodes(best_tree, t)


# %%
df_completo = pd.merge(df_with_leaf_nodes,df, left_index=True, right_index=True)
lista_colunas =  ['track_id', 'artists', 'album_name', 'track_name','popularity', 'duration_ms', 'explicit', 'danceability', 'leaf_node',
'energy','key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness','liveness', 'valence', 'tempo', 'time_signature', 'track_genre']
       
df_completo = df_completo[lista_colunas]

df_completo = df_completo.reset_index()

# %%
# def getLeafBased2DRecommendations(df, pc_cols=['PC1', 'PC2'],leaf_col='leaf_node'):
#    grouped = df.groupby('leaf_node')
#    leaf_node_groups = {leaf: group for leaf, group in df.groupby('leaf_node')}
#    for key in leaf_node_groups.keys():
#         sorted_group = .sort_values(by=['PC1', 'PC2'], ascending=[True, True])
   
from sklearn.decomposition import PCA


def addPCA(df, feature_start=5, feature_end=19, n_components=2):
    df = df.copy()
    X = df.iloc[:, feature_start:feature_end]
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    df['PC1'] = components[:, 0]
    df['PC2'] = components[:, 1]
    
    return df


def getLeafBased2DInfo(df, pc_cols=['PC1', 'PC2'], leaf_col='leaf_node'):
    # Group the dataframe by the leaf node
    grouped = df.groupby(leaf_col)
    
    # Create a dictionary with sorted groups
    leaf_node_groups = {}
    for leaf, group in grouped:
        sorted_group = group.sort_values(by=pc_cols, ascending=[True, True])
        leaf_node_groups[leaf] = sorted_group

    return grouped, leaf_node_groups

        
    
def add_directional_recommendations(df, leaf_node_groups, id_col='index', k=5):
    df = df.copy()
    df['Recommendations'] = None  # prepare column

    for leaf, sorted_leaf_df in leaf_node_groups.items():
        sorted_leaf_df = sorted_leaf_df.reset_index(drop=True)
        leaf_len = len(sorted_leaf_df)

        for idx, row in sorted_leaf_df.iterrows():
            song_id = row[id_col]

            # Determine direction of recommendation
            if idx < leaf_len / 2:
                recs = sorted_leaf_df.iloc[idx + 1:][id_col].tolist()
            else:
                recs = sorted_leaf_df.iloc[:idx][id_col].tolist()

            # Optional: limit number of recommendations
            if k is not None:
                recs = recs[:k]

            # Assign safely to the exact row in the main df
            df.at[df[df[id_col] == song_id].index[0], 'Recommendations'] = recs

    return df

            
        
    
df_completo= addPCA(df_completo)
a,b= getLeafBased2DInfo(df_completo)
directional_recommendations = add_directional_recommendations(df_completo,b)





# %%
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

def interactive_pca_plot(df, genre):
    genre_df = df[df["track_genre"] == genre].copy()

    features = genre_df.columns[5:19]  # Adjust if necessary
    X = genre_df[features]

    # === Run PCA ===
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # === Create DataFrame for plot ===
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["track_name"] = genre_df["track_name"].values
    pca_df["index"] = genre_df.index
    pca_df["track_genre"] = genre_df["track_genre"].values
    pca_df["origin"] = genre_df.index  # each track is its own origin
    pca_df["leaf_node"] = genre_df["leaf_node"].values
    pca_df["type"] = "original"

    # === Add recommendations ===
    for idx in genre_df.index:
        recommendations = df.loc[idx, "Recommendations"]
        for rec_idx in recommendations:
            rec_row = df.loc[rec_idx]
            rec_features = rec_row[features].values.reshape(1, -1)
            rec_pca = pca.transform(rec_features)
            pca_df = pd.concat([
                pca_df,
                pd.DataFrame([{
                    "PC1": rec_pca[0, 0],
                    "PC2": rec_pca[0, 1],
                    "track_name": rec_row["track_name"],
                    "leaf_node": rec_row["leaf_node"],
                    "index": rec_idx,
                    "track_genre": rec_row["track_genre"],
                    "origin": idx,
                    "type": "recommendation"
                }])
            ], ignore_index=True)

    # === Define colors based on original song ===
    unique_origins = pca_df[pca_df["type"] == "original"]["origin"].unique()
    color_map = {
        idx: px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
        for i, idx in enumerate(unique_origins)
    }
    pca_df["color"] = pca_df["origin"].map(color_map)

    # === Plot using only points with hover info ===
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="color",
        hover_data={
            "track_name": True,
            "index": True,
            "track_genre": True,
            "origin": True,
            "type": True,
            "leaf_node": True,
            "color": False,  # hide actual color code in hove
            "PC1": False,
            "PC2": False
        }
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8), showlegend=False)
    fig.update_layout(title=f"PCA of {genre} Tracks + Recommendations (Hover for details)")
    fig.show()
    return pca_df

interactive_pca_plot(directional_recommendations, "mpb")

# %%
interactive_pca_plot(directional_recommendations, "rock")

# %%
interactive_pca_plot(directional_recommendations, "death-metal")


