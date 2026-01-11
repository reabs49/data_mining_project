import numpy as np
import pandas as pd
import faiss
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

# ============================================================
# LOAD DATA
# ============================================================
data = pd.read_csv(r"D:\S3\data_mining\ML\data_set1.csv")
y = data["is_fire"].values
X = data.drop(columns=["is_fire"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

X_tr_fire = X_tr[y_tr == 1].reset_index(drop=True)
X_tr_nofire = X_tr[y_tr == 0].reset_index(drop=True)

X_train_fire = X_train[y_train == 1].reset_index(drop=True)
X_train_nofire = X_train[y_train == 0].reset_index(drop=True)

# ============================================================
# FAISS DISTANT POINT SELECTION
# ============================================================
def get_distant_points(X_df, n_points):
    if n_points >= len(X_df):
        return X_df.copy()

    X = np.ascontiguousarray(X_df.values.astype("float32"))
    faiss.normalize_L2(X)

    selected = [np.random.randint(len(X))]
    dist = 1 - X @ X[selected[0]]

    for _ in range(1, n_points):
        idx = np.argmax(dist)
        selected.append(idx)
        new_dist = 1 - X @ X[idx]
        dist = np.minimum(dist, new_dist)

    return X_df.iloc[selected].reset_index(drop=True)

# ============================================================
# BUILD SUB-DATASETS (UNCHANGED LOGIC)
# ============================================================
def build_subdatasets(X_fire, X_nofire, p):

    knn_fire = NearestNeighbors(n_neighbors=20).fit(X_fire.values)
    knn_nofire = NearestNeighbors(n_neighbors=20).fit(X_nofire.values)

    # --- Dataset 1: Hard boundary ---
    _, idx_nf = knn_nofire.kneighbors(
        X_fire.values, n_neighbors=int(p["k1_nofire_per_fire"])
    )
    nf_neighbors = X_nofire.iloc[np.unique(idx_nf.flatten())]

    X1 = pd.concat([X_fire, nf_neighbors])
    y1 = np.concatenate([np.ones(len(X_fire)), np.zeros(len(nf_neighbors))])

    # --- Dataset 2: Prototype-based ---
    proto_nf = get_distant_points(X_nofire, int(p["n_distant_nofire"]))

    fire_list, nofire_list = [], []
    for _, proto in proto_nf.iterrows():
        _, idx_f = knn_fire.kneighbors([proto.values], int(p["k2_fire_per_proto"]))
        _, idx_nf = knn_nofire.kneighbors([proto.values], int(p["k3_nofire_per_proto"]))

        fire_list.append(X_fire.iloc[idx_f[0]])
        nofire_list.append(X_nofire.iloc[idx_nf[0]])

    X2_fire = pd.concat(fire_list).drop_duplicates()
    X2_nofire = pd.concat(nofire_list).drop_duplicates()

    X2 = pd.concat([X2_fire, X2_nofire])
    y2 = np.concatenate([np.ones(len(X2_fire)), np.zeros(len(X2_nofire))])

    # --- Dataset 3: Global ---
    global_nf = get_distant_points(X_nofire, int(p["n_general_nofire"]))
    X3 = pd.concat([X_fire, global_nf])
    y3 = np.concatenate([np.ones(len(X_fire)), np.zeros(len(global_nf))])

    return (X1, y1), (X2, y2), (X3, y3)

# ============================================================
# TRAIN DECISION TREES (REGULARIZED)
# ============================================================
def train_trees(datasets, p):
    trees = []

    for i, (X, y) in enumerate(datasets):
        dt = DecisionTreeClassifier(
            max_depth=int(p["max_depth"]),
            min_samples_split=int(p["min_samples_split"]),
            min_samples_leaf=int(p["min_samples_leaf"]),
            random_state=42 + i
        )
        dt.fit(X, y)
        trees.append(dt)

    return trees

def predict_prob_ensemble(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

# ============================================================
# BAYESIAN SEARCH SPACE
# ============================================================
space = [
    Integer(3, 30, name="k1_nofire_per_fire"),
    Integer(2, 10, name="k2_fire_per_proto"),
    Integer(5, 30, name="k3_nofire_per_proto"),
    Integer(200, 900, name="n_distant_nofire"),
    Integer(2000, 9000, name="n_general_nofire"),
    Integer(6, 18, name="max_depth"),
    Integer(10, 50, name="min_samples_split"),
    Integer(5, 20, name="min_samples_leaf"),
    Real(0.05, 0.7, name="threshold"),
    Real(0.3, 0.8, name="alpha")
]

# ============================================================
# OBJECTIVE FUNCTION
# ============================================================
recall_fire_list = []
recall_nofire_list = []

@use_named_args(space)
def objective(**p):

    start = time.time()

    datasets = build_subdatasets(X_tr_fire, X_tr_nofire, p)
    trees = train_trees(datasets, p)

    y_val_prob = predict_prob_ensemble(trees, X_val)
    y_val_pred = (y_val_prob >= p["threshold"]).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    recall_fire = tp / (tp + fn + 1e-9)
    recall_nofire = tn / (tn + fp + 1e-9)

    recall_fire_list.append(recall_fire)
    recall_nofire_list.append(recall_nofire)

    score = p["alpha"] * recall_fire + (1 - p["alpha"]) * recall_nofire

    print(
        f"Recall_fire={recall_fire:.4f} | Recall_nofire={recall_nofire:.4f} | "
        f"Score={score:.4f} | thr={p['threshold']:.2f} | alpha={p['alpha']:.2f} | "
        f"time={time.time() - start:.1f}s"
    )

    return -score

# ============================================================
# RUN BAYESIAN OPTIMIZATION
# ============================================================
result = gp_minimize(
    objective,
    space,
    n_calls=60,
    random_state=42,
    verbose=True
)

# ============================================================
# SAVE RESULTS
# ============================================================
results_df = pd.DataFrame(result.x_iters, columns=[s.name for s in space])
results_df["score"] = -np.array(result.func_vals)
results_df["recall_fire"] = recall_fire_list
results_df["recall_nofire"] = recall_nofire_list

results_df_sorted = results_df.sort_values("score", ascending=False)
results_df_sorted.to_csv(
    "bayes_results_dt_sorted.csv", index=False
)

# ============================================================
# RESTORE BEST PARAMETERS
# ============================================================
INT_PARAMS = [
    "k1_nofire_per_fire", "k2_fire_per_proto", "k3_nofire_per_proto",
    "n_distant_nofire", "n_general_nofire",
    "max_depth", "min_samples_split", "min_samples_leaf"
]
FLOAT_PARAMS = ["threshold", "alpha"]

raw_best = results_df_sorted.iloc[0].to_dict()
best_params = {}

for k in INT_PARAMS:
    best_params[k] = int(round(raw_best[k]))
for k in FLOAT_PARAMS:
    best_params[k] = float(raw_best[k])

print("\nBEST PARAMETERS:")
for k, v in best_params.items():
    print(f"{k}: {v}")

# ============================================================
# FINAL TRAINING
# ============================================================
final_datasets = build_subdatasets(X_train_fire, X_train_nofire, best_params)
final_trees = train_trees(final_datasets, best_params)

# ============================================================
# TEST EVALUATION
# ============================================================
y_test_prob = predict_prob_ensemble(final_trees, X_test)
y_test_pred = (y_test_prob >= best_params["threshold"]).astype(int)

print("\nCONFUSION MATRIX (TEST)")
print(confusion_matrix(y_test, y_test_pred))

print("\nCLASSIFICATION REPORT (TEST)")
print(classification_report(y_test, y_test_pred, digits=4))

print("\nROC-AUC:", roc_auc_score(y_test, y_test_prob))

# ============================================================
# SAVE FINAL MODEL
# ============================================================
joblib.dump(
    {
        "trees": final_trees,
        "best_params": best_params,
        "threshold": best_params["threshold"],
        "alpha": best_params["alpha"]
    },
    "new_fire_detection_dt_recall_alpha_optimized.pkl"
)

print("\nModel saved as fire_detection_dt_recall_alpha_optimized.pkl")
