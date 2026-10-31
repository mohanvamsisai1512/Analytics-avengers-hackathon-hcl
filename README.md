# Analytics-avengers-hackathon-hcl
# Customer Churn Prediction and Retention Prioritization

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Losing customers affects recurring revenue, increases customer acquisition costs, and reduces long-term business growth.

This project develops an **AI-powered Customer Churn Prediction and Retention Prioritization System** that predicts the likelihood of customers churning within the next 30 days and helps businesses identify customers who require immediate attention.

The solution uses Machine Learning algorithms to analyze customer behaviour, subscription details, payment history, usage patterns, and satisfaction levels. It also uses SHAP explainability to understand the factors influencing churn predictions.

Beyond predicting churn, the project prioritizes customers based on their predicted churn probability and estimated business value, enabling targeted retention strategies.

## Business Problem

Subscription-based businesses often struggle to identify customers who are likely to discontinue their services.

Common warning signs include:

- Declining customer engagement and usage.
- Frequent failed payments.
- Poor customer satisfaction.
- Unresolved customer support tickets.
- Upcoming subscription renewals.
- Expiring promotional discounts.

Traditional retention strategies are often reactive, addressing customers only after they decide to leave.

**Objective:** Predict customers who are likely to churn within the next 30 days and prioritize them for proactive retention campaigns.

## Proposed Solution

The project follows an end-to-end Machine Learning workflow:

1. Data collection and exploratory data analysis.
2. Data cleaning and preprocessing.
3. Feature engineering to extract meaningful customer behaviour patterns.
4. Train-validation data splitting.
5. Model training using multiple classification algorithms.
6. Hyperparameter optimization using GridSearchCV.
7. Model evaluation using classification metrics.
8. Model explainability using SHAP.
9. Customer risk segmentation and retention prioritization.

## Dataset Description

The project uses a synthetic customer churn dataset containing:

- **Records:** 5,000 customers
- **Original features:** 30 columns
- **Target variable:** `churn_30d`

### Feature Categories

| Category | Description |
|---|---|
| Customer Profile | Customer ID, region, customer segment |
| Subscription Details | Plan tier, contract type, tenure |
| Customer Engagement | Usage hours, login frequency, last activity |
| Customer Satisfaction | Satisfaction score, NPS, support tickets |
| Payment Behaviour | Payment method, failed payments, on-time payments |
| Renewal Information | Days to renewal, discount expiry |
| Additional Services | Add-on services and subscription features |

### Target Variable

`churn_30d` is the binary classification target.

- `0` – Customer does not churn within 30 days.
- `1` – Customer churns within 30 days.

**Note:** The dataset is synthetic and intended for model development and demonstration. It does not represent actual customer records from a real company.

## Feature Engineering

New features are created from existing customer attributes to improve the model's ability to identify churn patterns.

| Engineered Feature | Description |
|---|---|
| `usage_per_active_day` | Average usage hours per login day |
| `engagement_decline_score` | Combined indicator of declining usage and inactivity |
| `payment_success_rate` | Proportion of payments made on time |
| `payment_failure_rate` | Failed payment proportion |
| `unresolved_ticket_ratio` | Proportion of unresolved support tickets |
| `renewal_discount_risk` | Identifies customers approaching renewal with expiring discounts |
| `downgrade_usage_decline` | Identifies customers who downgraded and reduced usage |

These features help capture customer behaviour, engagement, payment reliability, and potential subscription risks.

## Exploratory Data Analysis (EDA)

EDA is performed to understand customer behaviour and identify patterns associated with churn.

Key analysis includes:

- Dataset dimensions and data types.
- Missing-value and duplicate analysis.
- Target variable distribution.
- Numerical feature distributions.
- Box plots comparing churned and non-churned customers.
- Categorical feature analysis using cross-tabulation.
- Correlation heatmap.
- Numerical feature comparison across churn categories.

## Data Preprocessing

The following preprocessing techniques are used:

**Numerical Features**
- Missing-value imputation using median.
- Feature scaling using StandardScaler.

**Categorical Features**
- Missing-value imputation using the most frequent category.
- One-Hot Encoding to convert categorical variables into numerical representations.

Scikit-learn's Pipeline and ColumnTransformer are used to organize preprocessing.

Preprocessing parameters are fitted on training data only to prevent data leakage.

## Machine Learning Models

Three classification algorithms are compared.

### 1. Logistic Regression
- Used as a baseline classification model.
- Estimates customer churn probabilities.
- Uses regularization to control model complexity.

### 2. Random Forest
- Ensemble learning algorithm based on multiple decision trees.
- Captures nonlinear relationships.
- Helps identify complex customer behaviour patterns.

### 3. XGBoost
- Gradient boosting algorithm.
- Builds decision trees sequentially.
- Corrects errors from previous trees.
- Captures complex interactions between customer features.

## Hyperparameter Optimization

GridSearchCV is used to identify suitable hyperparameter configurations.

| Model | Hyperparameters |
|---|---|
| Logistic Regression | C, solver |
| Random Forest | n_estimators, max_depth, min_samples_split, min_samples_leaf |
| XGBoost | n_estimators, max_depth, learning_rate, subsample |

Cross-validation is used to compare different configurations.

PR-AUC (Average Precision) is selected as the primary optimization metric.

## Model Evaluation

The models are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| PR-AUC | Evaluates precision-recall performance |
| ROC-AUC | Measures the model's ability to distinguish between churn classes |
| Precision
