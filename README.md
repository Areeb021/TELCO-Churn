# Telco Customer Churn — Clustering, Regression & Predictive Modeling
 
## Problem Statement
Telecom companies lose revenue when customers cancel their subscriptions (churn). This project analyzes a telecom customer dataset to (1) segment customers into natural behavioral groups, (2) understand what drives monthly billing, and (3) predict which customers are likely to churn — so a retention team can prioritize outreach before customers leave.
 
## Dataset
- **Source:** IBM Telco Customer Churn dataset (7,043 customers, 21 features)
- **Features:** demographics (gender, senior citizen status, partner/dependents), account info (tenure, contract type, payment method, monthly/total charges), and subscribed services (internet, phone, streaming, tech support, etc.)
- **Target:** `Churn` (Yes/No)
## Approach
 
### 1. Data Cleaning
- Dropped `customerID` (no predictive value)
- Fixed `TotalCharges`, which was stored as text with blank values, by converting to numeric and handling missing entries
- Encoded binary columns (`gender`, `Churn`) with simple 0/1 mapping
- One-hot encoded multi-category columns (`Contract`, `PaymentMethod`, `InternetService`, etc.) to avoid implying false ordinal relationships
### 2. Unsupervised Exploration — K-Means + PCA
- Applied PCA to reduce the high-dimensional, one-hot-encoded feature space to 2 components for visualization
- Ran K-Means clustering on the PCA-reduced data to segment customers into behavioral groups (without using the `Churn` label)
- Compared churn rate and average tenure/monthly charges across clusters to identify which segments carry the highest churn risk
### 3. Regularized Regression — Ridge & Lasso
- Predicted `MonthlyCharges` from service and account features using Linear, Ridge, and Lasso Regression
- Found `MonthlyCharges` to be highly predictable (R² ≈ 0.999) since pricing is largely additive based on subscribed services — this confirms the models correctly learned the telecom's pricing structure, though it's a simpler prediction task than churn itself
### 4. Churn Classification — Logistic Regression & Random Forest
- Trained two classifiers to predict `Churn`, evaluated with `classification_report`, `confusion_matrix`, and ROC-AUC
- Compared models across **all decision thresholds** (not just the default 0.5) using ROC-AUC, since the right threshold is a business decision (e.g., how many customers retention teams can realistically follow up with)
- Both models performed comparably, with AUC scores in the ~0.84 range — confirming the models are meaningfully better than random guessing
- Extracted feature importances to identify the strongest churn predictors (e.g., contract type, tenure)
## Key Findings
- *(Fill in once feature importance and cluster outputs are reviewed — e.g., "Customers on month-to-month contracts with low tenure show the highest churn risk, concentrated in Cluster X.")*
- Logistic Regression and Random Forest achieved comparable predictive performance (AUC ~0.84), with [contract type / tenure / monthly charges] emerging as the strongest predictors of churn
- Clustering revealed [N] distinct customer segments with notably different churn rates, useful for targeted retention strategy
## Tech Stack
`Python` · `pandas` · `scikit-learn` · `matplotlib`
 
**Models used:** K-Means, PCA, Logistic Regression, Random Forest, Ridge Regression, Lasso Regression
 
## How to Run
```bash
pip install pandas scikit-learn matplotlib
python churn_analysis.py
```
 
## Project Structure
```
├── data.csv                # Telco Customer Churn dataset
├── churn_analysis.py       # Full pipeline: cleaning, clustering, regression, classification
└── README.md
```
 
## Future Improvements
- Hyperparameter tuning via GridSearchCV
- DBSCAN as an alternative clustering approach to compare against K-Means
- Deploy the churn classifier as a simple API or dashboard for non-technical retention teams
