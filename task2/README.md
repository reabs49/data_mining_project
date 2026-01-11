# Task 2 — supervised learning

## 📌 Overview

This directory contains the **supervised learning phase** of the *Data Mining Project*.  
It focuses on training, testing, and evaluating machine learning models using the dataset prepared in **Task 1**.

---

## 📁 Folder Contents

| File | Description |
|------|-------------|
| `random_forest_on_original_data.py` | scikit_learn Random Forest model trained on the original dataset (500K row) to do comparaison|
| `maybe_final16.py` | the customized one (RF) on the (500K row) -> Annex (from page 51) |
| `test1_3DT.py` | another customized (Decision trees) |
| `test.py` | re-resting the customized (RF) with prameters |
| `investigate.ipynb` | here we will use the results of the bayesian optimization (the best ones) saved in a csv file and select a kinda good ones |
| `travail.ipynb` | you will also find a file named work.ipynb in the task3 directory that includes the KNN and decision trees and random forest used on the (58K row data) for both scikit_learn and from scratch|

---


⚠️⚠️⚠️⚠️⚠️
for `maybe_final16.py` and `test1_3DT.py` the best results of the bayesian optimization are extracted in the 'investigate.ipynb' file 

## 🧠 Task Objective

The objectives of **Task 2** are to:

- Train machine learning models on the prepared dataset
- Evaluate baseline performance on original data
- Experiment with different modeling strategies
- Analyze performance under class imbalance conditions
- Prepare results for comparison and reporting

---
