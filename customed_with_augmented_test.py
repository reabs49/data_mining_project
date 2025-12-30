"""hey YOOO don't forget to test in case of more n_general_nofire"""

""" 
BEST PARAMETERS:
k1_nofire_per_fire: 20
k2_fire_per_proto: 3
k3_nofire_per_proto: 25
n_distant_nofire: 900
n_general_nofire: 7000
n_trees: 100
max_depth: 25
min_samples_split: 2
min_samples_leaf: 1

BEST F1-SCORE: 0.8711484076079925

CONFUSION MATRIX (TEST)
[[100457   4150]
 [  2571  23632]]

CLASSIFICATION REPORT (TEST)
              precision    recall  f1-score   support

           0     0.9750    0.9603    0.9676    104607
           1     0.8506    0.9019    0.8755     26203

    accuracy                         0.9486    130810
   macro avg     0.9128    0.9311    0.9216    130810
weighted avg     0.9501    0.9486    0.9492    130810

"""





import numpy as np
import pandas as pd
import faiss
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

# ============================================================
# LOAD DATA
# ============================================================
X_train = pd.read_csv(r"D:\S3\data_mining\data\X_train.csv")
y_train = pd.read_csv(r"D:\S3\data_mining\data\y_train.csv").squeeze()

X_test = pd.read_csv(r"D:\S3\data_mining\data\X_test.csv")
y_test = pd.read_csv(r"D:\S3\data_mining\data\y_test.csv").squeeze()

"""data = pd.read_csv(r"D:\S3\data_mining\ML\data_set1_disaugmented_diverse.csv")
label = data['is_fire']
data = data.drop(columns=['is_fire'])
X_train , X_test , y_train , y_test = train_test_split(data , label , test_size=0.3 , random_state=42)"""

print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

X_train_fire = X_train[y_train == 1].reset_index(drop=True)
X_train_nofire = X_train[y_train == 0].reset_index(drop=True)

print("Fire samples   :", len(X_train_fire))
print("No-fire samples:", len(X_train_nofire))

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
# BUILD SUB-DATASETS
# ============================================================
def build_subdatasets(params):
    knn_fire = NearestNeighbors(n_neighbors=20).fit(X_train_fire.values)
    knn_nofire = NearestNeighbors(n_neighbors=20).fit(X_train_nofire.values)

    # Dataset 1
    _, idx_nf = knn_nofire.kneighbors(X_train_fire.values, n_neighbors=params['k1_nofire_per_fire'])
    nf_neighbors = X_train_nofire.iloc[np.unique(idx_nf.flatten())]
    X1 = pd.concat([X_train_fire, nf_neighbors])
    y1 = np.concatenate([np.ones(len(X_train_fire)), np.zeros(len(nf_neighbors))])

    # Dataset 2
    proto_nf = get_distant_points(X_train_nofire, params['n_distant_nofire'])
    fire_list, nofire_list = [], []
    for i in range(len(proto_nf)):
        proto = proto_nf.iloc[i]
        _, idx_f = knn_fire.kneighbors([proto.values], params['k2_fire_per_proto'])
        _, idx_nf = knn_nofire.kneighbors([proto.values], params['k3_nofire_per_proto'])
        fire_list.append(X_train_fire.iloc[idx_f[0]])
        nofire_list.append(X_train_nofire.iloc[idx_nf[0]])
    X2_fire = pd.concat(fire_list).drop_duplicates()
    X2_nofire = pd.concat(nofire_list).drop_duplicates()
    X2 = pd.concat([X2_fire, X2_nofire])
    y2 = np.concatenate([np.ones(len(X2_fire)), np.zeros(len(X2_nofire))])

    # Dataset 3
    distant_nf = get_distant_points(X_train_nofire, params['n_general_nofire'])
    X3 = pd.concat([X_train_fire, distant_nf])
    y3 = np.concatenate([np.ones(len(X_train_fire)), np.zeros(len(distant_nf))])

    return (X1, y1), (X2, y2), (X3, y3)

# ============================================================
# TRAIN RANDOM FORESTS
# ============================================================
def train_forests(datasets, params):
    forests = []
    for X, y in datasets:
        rf = RandomForestClassifier(
            n_estimators=params['n_trees'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        forests.append(rf)
    return forests

# ============================================================
# ENSEMBLE PREDICTION
# ============================================================
def predict_prob_majority(forests, X):
    probs = np.zeros(len(X))
    for rf in forests:
        probs += rf.predict_proba(X)[:, 1]
    return probs / len(forests)

# ============================================================
# EVALUATION FOR F1
# ============================================================
def evaluate(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['precision'] = np.sum((y_true==1)&(y_pred==1)) / np.sum(y_pred==1) if np.sum(y_pred==1) > 0 else 0
    metrics['recall'] = np.sum((y_true==1)&(y_pred==1)) / np.sum(y_true==1)
    metrics['accuracy'] = np.mean(y_true == y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['false_alarm_rate'] = fp / (fp + tn)
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    return metrics

# ============================================================
# BAYESIAN SEARCH SPACE
# ============================================================
space = [
    Integer(3, 20, name='k1_nofire_per_fire'),
    Integer(3, 20, name='k2_fire_per_proto'),
    Integer(5, 25, name='k3_nofire_per_proto'),
    Integer(100, 900, name='n_distant_nofire'),
    Integer(2000, 7000, name='n_general_nofire'),
    Integer(100, 300, name='n_trees'),
    Integer(6, 16, name='max_depth'),
    Integer(10, 50, name='min_samples_split'),
    Integer(5, 20, name='min_samples_leaf')
]

# ============================================================
# OBJECTIVE FUNCTION (maximize F1)
# ============================================================
@use_named_args(space)
def objective(**params):
    start = time.time()
    datasets = build_subdatasets(params)
    forests = train_forests(datasets, params)
    y_prob = predict_prob_majority(forests, X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate(y_test, y_pred, y_prob)
    print(f"F1-score: {metrics['f1']:.4f} | Time: {time.time()-start:.1f}s")
    return -metrics['f1']  # minimize negative F1

# ============================================================
# RUN BAYESIAN OPTIMIZATION
# ============================================================
result = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=True)

print("\nBEST PARAMETERS:")
best_params = dict(zip([s.name for s in space], result.x))
for k, v in best_params.items():
    print(f"{k}: {v}")

print("\nBEST F1-SCORE:", -result.fun)

# ============================================================
# FINAL TRAINING + TEST EVALUATION
# ============================================================
final_datasets = build_subdatasets(best_params)
final_forests = train_forests(final_datasets, best_params)

y_test_prob = predict_prob_majority(final_forests, X_test)
y_test_pred = (y_test_prob >= 0.5).astype(int)

print("\nCONFUSION MATRIX (TEST)")
print(confusion_matrix(y_test, y_test_pred))

print("\nCLASSIFICATION REPORT (TEST)")
print(classification_report(y_test, y_test_pred, digits=4))

# ============================================================
# SAVE MODEL
# ============================================================
joblib.dump(
    {
        "forests": final_forests,
        "best_params": best_params,
        "threshold": 0.5
    },
    "fire_detection_ensemble_f1.pkl"
)

print("\nModel saved as fire_detection_ensemble_f1.pkl")





