import numpy as np
import pandas as pd
import faiss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    accuracy_score
)

# ============================================================
# LOAD DATA
# ============================================================
data = pd.read_csv(r"D:\S3\data_mining\ML\data_set1.csv")
y = data["is_fire"]
X = data.drop(columns=["is_fire"])

# Train/Validation/Test split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

# Fire / No-fire splits
X_train_fire = X_train[y_train == 1].reset_index(drop=True)
X_train_nofire = X_train[y_train == 0].reset_index(drop=True)

X_val_fire = X_val[y_val == 1].reset_index(drop=True)
X_val_nofire = X_val[y_val == 0].reset_index(drop=True)

X_test_fire = X_test[y_test == 1].reset_index(drop=True)
X_test_nofire = X_test[y_test == 0].reset_index(drop=True)

# ============================================================
# PARAMETERS (best found)
# ============================================================
best_params = {
    "k1_nofire_per_fire": 30,
    "k2_fire_per_proto": 10,
    "k3_nofire_per_proto": 30,
    "n_distant_nofire": 900,
    "n_general_nofire": 9000, 
    "n_trees": 300,
    "max_depth": 18,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "threshold": 0.29193978641207163,
    "alpha": 0.7095761725818651
}

# ============================================================
# FAISS DISTANT POINT SELECTION
# ============================================================
def get_distant_points(X_df, n_points):
    if n_points >= len(X_df):
        return X_df.copy()
    X_arr = np.ascontiguousarray(X_df.values.astype("float32"))
    faiss.normalize_L2(X_arr)
    selected = [np.random.randint(len(X_arr))]
    dist = 1 - X_arr @ X_arr[selected[0]]
    for _ in range(1, n_points):
        idx = np.argmax(dist)
        selected.append(idx)
        new_dist = 1 - X_arr @ X_arr[idx]
        dist = np.minimum(dist, new_dist)
    return X_df.iloc[selected].reset_index(drop=True)

# ============================================================
# BUILD SUB-DATASETS
# ============================================================
def build_subdatasets(X_fire, X_nofire, p):
    knn_fire = NearestNeighbors(n_neighbors=20).fit(X_fire.values)
    knn_nofire = NearestNeighbors(n_neighbors=20).fit(X_nofire.values)

    # Dataset 1
    _, idx_nf = knn_nofire.kneighbors(X_fire.values, n_neighbors=p["k1_nofire_per_fire"])
    nf_neighbors = X_nofire.iloc[np.unique(idx_nf.flatten())]
    X1 = pd.concat([X_fire, nf_neighbors])
    y1 = np.concatenate([np.ones(len(X_fire)), np.zeros(len(nf_neighbors))])

    # Dataset 2
    proto_nf = get_distant_points(X_nofire, p["n_distant_nofire"])
    fire_list, nofire_list = [], []
    for _, proto in proto_nf.iterrows():
        _, idx_f = knn_fire.kneighbors([proto.values], p["k2_fire_per_proto"])
        _, idx_nf = knn_nofire.kneighbors([proto.values], p["k3_nofire_per_proto"])
        fire_list.append(X_fire.iloc[idx_f[0]])
        nofire_list.append(X_nofire.iloc[idx_nf[0]])
    X2_fire = pd.concat(fire_list).drop_duplicates()
    X2_nofire = pd.concat(nofire_list).drop_duplicates()
    X2 = pd.concat([X2_fire, X2_nofire])
    y2 = np.concatenate([np.ones(len(X2_fire)), np.zeros(len(X2_nofire))])

    # Dataset 3
    distant_nf = get_distant_points(X_nofire, p["n_general_nofire"])
    X3 = pd.concat([X_fire, distant_nf])
    y3 = np.concatenate([np.ones(len(X_fire)), np.zeros(len(distant_nf))])

    return (X1, y1), (X2, y2), (X3, y3)

# ============================================================
# TRAIN RANDOM FORESTS
# ============================================================
def train_forests(datasets, p):
    forests = []
    for i, (X, y) in enumerate(datasets):
        rf = RandomForestClassifier(
            n_estimators=p["n_trees"],
            max_depth=p["max_depth"],
            min_samples_split=p["min_samples_split"],
            min_samples_leaf=p["min_samples_leaf"],
            class_weight="balanced",
            random_state=42+i,
            n_jobs=-1
        )
        rf.fit(X, y)
        forests.append(rf)
    return forests

def predict_prob_ensemble(forests, X):
    return np.mean([rf.predict_proba(X)[:, 1] for rf in forests], axis=0)

# ============================================================
# EVALUATE METRICS
# ============================================================
def evaluate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall_fire = tp / (tp + fn)
    recall_nofire = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    return {
        "recall_fire": recall_fire,
        "recall_nofire": recall_nofire,
        "f1": f1,
        "precision": precision,
        "accuracy": accuracy,
        "roc_auc": roc_auc
    }

# ============================================================
# TRAIN & PREDICT
# ============================================================
datasets_train = build_subdatasets(X_train_fire, X_train_nofire, best_params)
forests = train_forests(datasets_train, best_params)

# Train metrics
y_train_prob = predict_prob_ensemble(forests, X_train)
y_train_pred = (y_train_prob >= best_params["threshold"]).astype(int)
train_metrics = evaluate_metrics(y_train, y_train_pred, y_train_prob)

# Validation metrics
y_val_prob = predict_prob_ensemble(forests, X_val)
y_val_pred = (y_val_prob >= best_params["threshold"]).astype(int)
val_metrics = evaluate_metrics(y_val, y_val_pred, y_val_prob)

# Test metrics
y_test_prob = predict_prob_ensemble(forests, X_test)
y_test_pred = (y_test_prob >= best_params["threshold"]).astype(int)
test_metrics = evaluate_metrics(y_test, y_test_pred, y_test_prob)

# ============================================================
# PRINT RESULTS
# ============================================================
def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

print_metrics("TRAIN SET", train_metrics)
print_metrics("VALIDATION SET", val_metrics)
print_metrics("TEST SET", test_metrics)
