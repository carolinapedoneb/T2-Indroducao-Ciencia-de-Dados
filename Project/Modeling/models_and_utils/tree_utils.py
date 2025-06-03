from sklearn.tree import _tree
import pandas as pd

def extract_rules(tree, feature_names, X_data, index_data):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, depth, data_indices):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            entropy = tree_.impurity[node]
            left = tree_.children_left[node]
            right = tree_.children_right[node]
            rules.append({
                "feature": name,
                "threshold": threshold,
                "": entropy,
                "depth": depth,
                "node": node,
                "data_indices": data_indices
            })
            left_indices = data_indices[X_data.loc[data_indices, name] <= threshold]
            right_indices = data_indices[X_data.loc[data_indices, name] > threshold]
            recurse(left, depth + 1, left_indices)
            recurse(right, depth + 1, right_indices)
    
    recurse(0, 0, index_data)
    return rules


def is_leaf(tree, node_id):
    """Check if a node is a leaf."""
    return (
        tree.children_left[node_id] == _tree.TREE_LEAF
        and tree.children_right[node_id] == _tree.TREE_LEAF
    )

def get_leaf_nodes(tree):
    """Get list of all leaf node indices in the tree."""
    leaf_indices = []
    for node_id in range(tree.node_count):
        if is_leaf(tree, node_id):
            leaf_indices.append(node_id)
    return leaf_indices

def get_samples_leaf_nodes(decision_tree, X_desired):
    """
    Apply trained decision_tree to X_desired and return a DataFrame mapping
    each sample to the leaf node it ends up in.
    """
    tree_structure = decision_tree.tree_
    node_indices = decision_tree.apply(X_desired)

    df_nodes = pd.DataFrame({
        "sample_index": X_desired.index,
        "leaf_node": node_indices
    })

    # Filter to include only true leaf nodes
    leaf_nodes = get_leaf_nodes(tree_structure)
    df_leaf_only = df_nodes[df_nodes["leaf_node"].isin(leaf_nodes)]

    return df_leaf_only
