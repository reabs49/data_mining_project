
# Task 1 — Data Preparation and Integration

## 📌 Overview

This directory corresponds to **step 1** of the *Data Mining Project*.  
The goal of this task is to **prepare, clean, and integrate multiple heterogeneous datasets** related to **fire events, climate variables, soil characteristics, elevation, and land cover**, in order to build a reliable dataset for subsequent analysis and modeling.

Task 1 focuses on:
- Data loading and exploration
- Cleaning and preprocessing
- Spatial and feature-level integration of multiple data sources

---

## 📁 Folder Contents

| File | Description |
|------|-------------|
| `task1(reading_exploring_clipping_fire_landcover_climate...).ipynb` | Initial data loading, exploration, and spatial clipping of fire, land cover, and climate data |
| `data_cleaning(fire_climate).ipynb` | Cleaning and preprocessing of fire and climate datasets |
| `climate_coordinates.ipynb` | Extraction and processing of climate data coordinates 2KM*2KM (and 7KM*7KM in a second time) and assigne fire presence or absence to each coordinate|
| `merge_climatefire_soil.ipynb` | Merging fire and climate data with soil information |
| `merge_climatefiresoil_elevation.ipynb` | Final integration step adding elevation data |
---

## 🧠 Task Objective

The main objectives of **Task 1** are:

- Understand the structure and quality of each dataset
- Handle missing values, inconsistencies, and noisy data
- Perform spatial alignment (coordinates, clipping, matching)
- Merge multiple data sources into a unified dataset
- Produce a clean dataset ready for modeling in later tasks

---
