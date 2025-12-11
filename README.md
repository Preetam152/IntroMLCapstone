# IntroMLCapstone
# IntroMLCapstone: House Price Prediction using Classic and Research-Based ML
## Project Overview

This capstone project addresses a real-world regression problem: predicting residential house prices using the Kaggle House Prices - Advanced Regression Techniques dataset. The goal is to compare the performance of classic machine learning algorithms against advanced approaches found in recent peer-reviewed literature.

The project encompasses a full machine learning workflow: in-depth data exploration and preprocessing, implementation of multiple regression models, hyperparameter tuning, and a final comparative analysis using industry-standard metrics (MSE, MAE, RMSLE).

## Dataset

 **Source:** Kaggle: House Prices - Advanced Regression Techniques
 (https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overviewLinks)
**Target Variable:** `SalePrice`
**Features:** 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa.

## Tasks Implemented

1. **Data Preprocessing:** Handled missing values (imputation), encoded categorical features (One-Hot Encoding), engineered new features, and applied log transformations to normalize skewed features, including the target variable.
2. **Classic ML Algorithms (Task 2):**
    * Ridge Regression (Linear Regularization)
    * Random Forest Regressor (Ensemble/Tree-based)
    * Support Vector Regressor (SVR)
3.  **Literature-Based Implementations (Task 3):**
     **Paper 1 Approach:** **Stacked Generalization Regression** (Ensemble Method)
        * *Concept:* Combining predictions from multiple diverse base models (Ridge, Lasso, ElasticNet, SVR, GBT) with a meta-model to minimize 
                     prediction errors.
     **Paper 2 Approach:** **Lasso Regression with Log Transformation and Feature Selection**
        * *Concept:* Applying strong L1 regularization to a heavily preprocessed dataset (log-transformed target) for feature selection and 
                     interpretability.
4. **Comparison Tables of Performances like MSE (Task 4):**
    * Comparison tables of performance like MSE
    * Plots (e.g., actual vs. predicted values, feature importances)
    * Interpretation of findings

    

## File Structure

| Filename | Description | Task(s) |
|`data_preprocessing.ipynb` | Core data cleaning, feature engineering, and feature selection. Saves processed data for all models.| 1 |
| `classic_model_ridge.ipynb` | Implementation and evaluation of the Ridge Regression model. | 2 |
| `classic_model_random_forest.ipynb` | Implementation and evaluation of the Random Forest Regressor. | 2 |
| `classic_model_svr.ipynb` | Implementation and evaluation of the Support Vector Regressor. | 2 |
| `literature_stacked_ensemble.ipynb` | Implementation and evaluation of the Stacking/Hybrid model. | 3 |
| `literature_lasso_regularization.ipynb` | Implementation and evaluation of the Lasso Regression model with a focus on regularization and log transformation. | 3 |
| `final_results_comparison.ipynb` | Aggregates metrics from all 5 models to generate the final comparison table and plots for the technical report. | 4 |

## Dependencies
1. numpy
2. pandas
3. scikit-learn
4. matplotlib
5. seaborn
