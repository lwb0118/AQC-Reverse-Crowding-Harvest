"""
ml_model.py
===========
Machine learning extension: predict future crowding unwind / reversal.
Uses logistic regression and tree-based models.
Note: Requires scikit-learn. This is a framework — actual performance
depends on data quality and feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


def prepare_ml_data(panel, feature_cols, target_col, min_samples=100):
    """
    Prepare data for ML prediction.
    
    Target: Y(i,t+1) = 1 if stock is in top 20% of future crowding (unwind)
            0 otherwise
    """
    df = panel.copy()
    
    # Create binary target: top 20% of crowding decay
    df['target'] = 0
    for d, grp in df.groupby('date'):
        if target_col in grp.columns:
            threshold = grp[target_col].quantile(0.8)
            df.loc[grp.index, 'target'] = (grp[target_col] > threshold).astype(int)
    
    # Features and target
    X = df[feature_cols].copy()
    y = df['target'].values
    
    # Handle missing values
    X = X.fillna(0)
    
    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Remove rows where target is NaN
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    
    return X, y


def run_ml_models(X, y, test_size=0.2):
    """
    Run logistic regression and random forest with time-series split.
    
    Parameters
    ----------
    X : pd.DataFrame
    y : np.array
    test_size : float
    
    Returns
    -------
    dict of model results
    """
    results = {}
    
    # Time-series split (respect temporal order)
    n = len(y)
    split_idx = int(n * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {'error': 'Insufficient class diversity for train/test split'}
    
    # 1. Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]
        
        results['logistic_regression'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_prob)),
            'coefs': dict(zip(X.columns, lr.coef_[0])),
        }
    except Exception as e:
        results['logistic_regression'] = {'error': str(e)}
    
    # 2. Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                     random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_prob = rf.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_prob)),
            'feature_importance': dict(
                sorted(zip(X.columns, rf.feature_importances_),
                       key=lambda x: x[1], reverse=True)
            ),
        }
    except Exception as e:
        results['random_forest'] = {'error': str(e)}
    
    # Cross-validation
    try:
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_cv_train = X.iloc[train_idx]
            X_cv_val = X.iloc[val_idx]
            y_cv_train = y[train_idx]
            y_cv_val = y[val_idx]
            
            lr_cv = LogisticRegression(max_iter=1000, random_state=42)
            lr_cv.fit(X_cv_train, y_cv_train)
            cv_scores.append(float(lr_cv.score(X_cv_val, y_cv_val)))
        
        results['cv_logistic'] = {
            'mean_accuracy': float(np.mean(cv_scores)),
            'std_accuracy': float(np.std(cv_scores, ddof=1)),
            'scores': cv_scores,
        }
    except Exception as e:
        results['cv_logistic'] = {'error': str(e)}
    
    return results
