# E-commerce Customer Segmentation

## What This Project Is About (Big Picture)

In this project, we are building an **end-to-end unsupervised machine learning system** to:

- Segment e-commerce customers based on their purchasing behavior
- Identify anomalous / unusual customers separately

This mimics a real-world business analytics + ML deployment workflow, not just a notebook experiment.

---

## Why This Project Exists (Problem Statement)

E-commerce platforms have thousands of customers, but:

* Not all customers behave the same
* Treating everyone equally wastes marketing budget
* Rare or abnormal customers distort segmentation results

**Goal:**
Use unsupervised learning to:

* Group similar customers together
* Detect customers that don't fit normal patterns

---

## What Data We Use

We use **real transactional data** from an online retail store.

Each row represents a **transaction**, not a customer.

So we:

* Convert raw transactions into **customer-level behavior**
* Engineer meaningful features like:

  * How recently a customer purchased
  * How often they purchase
  * How much they spend

This step reflects real industry practice.

---

## What We Actually Do (Step-by-Step)

### 1. Data Understanding

* Explore raw transaction data
* Identify missing values and inconsistencies
* Decide what data is relevant for customer analysis

---

### 2. Feature Engineering (Core of the Project)

We transform transaction data into **customer behavior metrics** such as:

* Recency
* Frequency
* Monetary value
* Average order value
* Customer tenure

- This converts raw data into **ML-ready features**

---

### 3. Data Preprocessing

* Handle missing values
* Remove invalid transactions
* Treat outliers
* Scale features (important for clustering)

---

### 4. Exploratory Data Analysis (EDA)

* Understand customer spending patterns
* Analyze relationships between features
* Identify skewness and outliers
* Derive business insights, not just plots

---

### 5. Unsupervised Modeling

#### K-Means Clustering

* Used for main customer segmentation
* Helps identify:

  * Low-value customers
  * Medium-value customers
  * High-value customers

#### DBSCAN

* Used for anomaly detection
* Identifies:

  * Rare customers
  * Extremely high spenders
  * Unusual purchasing behavior

- DBSCAN is used in addition to, not instead of, K-Means.

---

### 6. Model Evaluation & Tuning

* Elbow method
* Silhouette score
* PCA visualization
* DBSCAN parameter tuning (`eps`, `min_samples`)

---

### 7. Deployment (Flask API)

We expose the trained models via a Flask REST API that:

* Accepts customer behavior metrics as input
* Returns:

  * Customer segment (K-Means)
  * Anomaly flag (DBSCAN)

This makes the project production-oriented, not just academic.

---

## What This Project Demonstrates

This project shows skills in:

* Unsupervised learning
* Feature engineering
* Data preprocessing
* Clustering evaluation
* Anomaly detection
* Model deployment using Flask
* Clean project structuring

---