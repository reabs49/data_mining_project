import numpy as np
import pandas as pd
import faiss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_recall_curve, auc, roc_auc_score
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
import time

print("="*70)
print("RF ENSEMBLE + BAYESIAN OPTIMIZATION WITH PR-AUC")
print("="*70)

# ============================================================
# 1. LOAD DATA
# ============================================================
data = pd.read_csv(r"D:\S3\data_mining\ML\data_set2_disaugmented_diverse.csv")

X = data.drop(columns=["is_fire", "longitude", "latitude", "row", "col"])
y = data["is_fire"]

print(f"Total: {len(data)} samples")
print(f"Fire: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
print(f"No-fire: {(~y.astype(bool)).sum()}")

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split by class
X_fire = X_scaled[y == 1].reset_index(drop=True)
X_no_fire = X_scaled[y == 0].reset_index(drop=True)

x_train_f, x_test_f = train_test_split(X_fire, test_size=0.3, random_state=42)
x_train_nf, x_test_nf = train_test_split(X_no_fire, test_size=0.3, random_state=42)

x_test = pd.concat([x_test_f, x_test_nf]).reset_index(drop=True)
y_test = np.concatenate([np.ones(len(x_test_f)), np.zeros(len(x_test_nf))])

print(f"\nTrain: {len(x_train_f)} fire, {len(x_train_nf)} no-fire")
print(f"Test: {len(x_test_f)} fire, {len(x_test_nf)} no-fire")

# ============================================================
# 2. HELPER: GET DISTANT POINTS
# ============================================================
def get_distant_points(X_df, n_points):
    if n_points >= len(X_df):
        return X_df
    X_np = np.ascontiguousarray(X_df.to_numpy().astype("float32"))
    faiss.normalize_L2(X_np)
    chosen = [np.random.randint(len(X_np))]
    dists = 1 - X_np @ X_np[chosen[0]]
    for _ in range(1, n_points):
        idx = np.argmax(dists)
        chosen.append(idx)
        new_dist = 1 - X_np @ X_np[idx]
        dists = np.minimum(dists, new_dist)
    return X_df.iloc[chosen].reset_index(drop=True)

# ============================================================
# 3. BUILD SUB-DATASETS
# ============================================================
def build_subdatasets(params):
    knn_fire = NearestNeighbors(n_neighbors=20).fit(x_train_f.values)
    knn_nofire = NearestNeighbors(n_neighbors=20).fit(x_train_nf.values)
    
    # Sub-dataset 1
    _, idx_nf = knn_nofire.kneighbors(x_train_f.values, n_neighbors=params['k1_nofire_per_fire'])
    nf_neighbors_1 = x_train_nf.iloc[np.unique(idx_nf.flatten())].reset_index(drop=True)
    X_1 = pd.concat([x_train_f, nf_neighbors_1]).reset_index(drop=True)
    y_1 = np.concatenate([np.ones(len(x_train_f)), np.zeros(len(nf_neighbors_1))])
    
    # Sub-dataset 2
    proto_nf = get_distant_points(x_train_nf, params['n_distant_nofire'])
    fire_list, nofire_list = [], []
    for i in range(len(proto_nf)):
        proto = proto_nf.iloc[i]
        _, idx_f = knn_fire.kneighbors([proto.values], n_neighbors=params['k2_fire_per_proto'])
        fire_list.append(x_train_f.iloc[idx_f[0]])
        _, idx_nf = knn_nofire.kneighbors([proto.values], n_neighbors=params['k3_nofire_per_proto'])
        nofire_list.append(x_train_nf.iloc[idx_nf[0]])
    X_2_f = pd.concat(fire_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
    X_2_nf = pd.concat(nofire_list, ignore_index=True).drop_duplicates().reset_index(drop=True)
    X_2 = pd.concat([X_2_f, X_2_nf]).reset_index(drop=True)
    y_2 = np.concatenate([np.ones(len(X_2_f)), np.zeros(len(X_2_nf))])
    
    # Sub-dataset 3
    distant_nf = get_distant_points(x_train_nf, params['n_general_nofire'])
    X_3 = pd.concat([x_train_f, distant_nf]).reset_index(drop=True)
    y_3 = np.concatenate([np.ones(len(x_train_f)), np.zeros(len(distant_nf))])
    
    return (X_1, y_1), (X_2, y_2), (X_3, y_3)

# ============================================================
# 4. TRAIN 3 RANDOM FORESTS
# ============================================================
def train_forests(datasets, params):
    forests = []
    for X, y in datasets:
        rf = RandomForestClassifier(
            n_estimators=params['n_trees'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        forests.append(rf)
    return forests

# ============================================================
# 5. PREDICT WITH PROBABILITIES (FOR PR-AUC)
# ============================================================
def predict_prob_majority(forests, X_test):
    prob_sum = np.zeros(len(X_test))
    for rf in forests:
        prob_sum += rf.predict_proba(X_test)[:, 1]
    prob_avg = prob_sum / len(forests)
    y_pred = (prob_avg >= 0.5).astype(int)
    return y_pred, prob_avg

# ============================================================
# 6. EVALUATION
# ============================================================
def evaluate(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fire_recall = tp / (tp + fn)
    fire_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_alarm_rate = fp / (tn + fp)
    
    return {
        'fire_recall': fire_recall,
        'fire_precision': fire_precision,
        'accuracy': accuracy,
        'false_alarm_rate': false_alarm_rate,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'cm': cm
    }

# ============================================================
# 7. BAYESIAN OPTIMIZATION
# ============================================================
space = [
    Integer(5, 15, name='k1_nofire_per_fire'),
    Integer(400, 800, name='n_distant_nofire'),
    Integer(4, 10, name='k2_fire_per_proto'),
    Integer(10, 20, name='k3_nofire_per_proto'),
    Integer(5000, 15000, name='n_general_nofire'),
    Integer(50, 150, name='n_trees'),
    Integer(10, 20, name='max_depth'),
    Integer(5, 20, name='min_samples_split'),
    Integer(3, 10, name='min_samples_leaf')
]

iteration = [0]
best_score = [-np.inf]
best_params = [None]

@use_named_args(space)
def objective(**params):
    iteration[0] += 1
    print(f"\n--- Iteration {iteration[0]} ---")
    try:
        datasets = build_subdatasets(params)
        forests = train_forests(datasets, params)
        _, y_prob = predict_prob_majority(forests, x_test)
        evals = evaluate(y_test, y_prob)
        score = evals['pr_auc']  # Optimize PR-AUC
        
        if score > best_score[0]:
            best_score[0] = score
            best_params[0] = params.copy()
            print(f"🎉 NEW BEST PR-AUC: {score:.4f}")
        
        return -score
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0

print("\nStarting Bayesian Optimization...")
start_time = time.time()
result = gp_minimize(objective, space, n_calls=30, n_initial_points=10, random_state=42)
elapsed = time.time() - start_time

# ============================================================
# 8. FINAL MODEL
# ============================================================
print("\nBest PR-AUC:", best_score[0])
print("Best Parameters:")
for k, v in best_params[0].items():
    print(f"  {k}: {v}")

datasets = build_subdatasets(best_params[0])
forests = train_forests(datasets, best_params[0])
y_pred, y_prob = predict_prob_majority(forests, x_test)
evals = evaluate(y_test, y_prob)

print("\nConfusion Matrix:\n", evals['cm'])
print(f"Fire Recall: {evals['fire_recall']:.4f}")
print(f"Fire Precision: {evals['fire_precision']:.4f}")
print(f"Accuracy: {evals['accuracy']:.4f}")
print(f"False Alarm Rate: {evals['false_alarm_rate']:.4f}")
print(f"PR-AUC: {evals['pr_auc']:.4f}")
print(f"ROC-AUC: {evals['roc_auc']:.4f}")
