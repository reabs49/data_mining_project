import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score

from skopt import BayesSearchCV
from skopt.space import Integer, Real


# =========================
# 1. Read data
# =========================
df = pd.read_csv(r"D:\S3\data_mining\ML\data_set2_disaugmented_diverse.csv")

# CHANGE 'target' to your label column name
X = df.drop(columns=['is_fire'])
y = df['is_fire']


# =========================
# 2. Train / Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 3. Random Forest model
# =========================
rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)


# =========================
# 4. Bayesian optimization space
# =========================
search_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(5, 40),
    'min_samples_split': Integer(2, 50),
    'min_samples_leaf': Integer(1, 20)
}


# =========================
# 5. Bayesian Optimization
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

bayes_search = BayesSearchCV(
    estimator=rf,
    search_spaces=search_space,
    n_iter=40,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1,
    random_state=42
)


# =========================
# 6. Fit Bayesian Search
# =========================
bayes_search.fit(X_train, y_train)


# =========================
# 7. Best parameters
# =========================
print("\nBEST PARAMETERS:")
for k, v in bayes_search.best_params_.items():
    print(f"{k}: {v}")

print("\nBEST CV ROC-AUC:", bayes_search.best_score_)


# =========================
# 8. Train final model
# =========================
best_rf = bayes_search.best_estimator_
best_rf.fit(X_train, y_train)


# =========================
# 9. Evaluation
# =========================
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, digits=4))

roc_auc = roc_auc_score(y_test, y_proba)
print("TEST ROC-AUC:", roc_auc)
