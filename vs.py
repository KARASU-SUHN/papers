import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# Assuming X_train, y_train, X_valid, y_valid, X_test are already defined

# Step 1: Apply One-Hot Encoding to the categorical variables for models that don't handle them natively
categorical_columns = ["line", "tray_no", "position", "maintenance_count"]
encoder = OneHotEncoder(drop='first', sparse=False)

# Fit and transform on training data
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])

# Apply transformation to the validation and test data
X_valid_encoded = encoder.transform(X_valid[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Drop original categorical columns and merge with the encoded columns for models that require one-hot encoding
X_train_oh = np.hstack((X_train.drop(columns=categorical_columns).values, X_train_encoded))
X_valid_oh = np.hstack((X_valid.drop(columns=categorical_columns).values, X_valid_encoded))
X_test_oh = np.hstack((X_test.drop(columns=categorical_columns).values, X_test_encoded))

# Step 2: Create a dictionary of classifiers, including CatBoost and LightGBM
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'K Neighbors': KNeighborsClassifier(),
    'SVC': SVC(probability=True),  # Enable probability output for SVC
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=1000),
    'XGB': XGBClassifier(eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(verbose=0),  # CatBoost can handle categorical variables directly
    'LightGBM': LGBMClassifier()  # LightGBM can handle categorical variables directly
}

# Step 3: Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation results
results = {}

# Perform k-fold cross-validation for each classifier
for name, clf in classifiers.items():
    if name in ['CatBoost', 'LightGBM']:
        # Use the raw categorical data for CatBoost and LightGBM
        preds = cross_val_predict(clf, X_train, y_train, cv=kf, method="predict_proba")
    else:
        # Use the one-hot encoded data for other classifiers
        preds = cross_val_predict(clf, X_train_oh, y_train, cv=kf, method="predict_proba")
    
    # Compute ROC AUC score using the second column (positive class probabilities)
    roc_auc = roc_auc_score(y_train, preds[:, 1])
    
    results[name] = roc_auc
    print(f"{name}: ROC AUC = {roc_auc:.4f}")

# Step 4: Train the best model and validate on the validation set (X_valid, y_valid)
for name, clf in classifiers.items():
    if name in ['CatBoost', 'LightGBM']:
        # Use raw categorical data for CatBoost and LightGBM
        clf.fit(X_train, y_train)
        preds_valid = clf.predict_proba(X_valid)
    else:
        # Use one-hot encoded data for the other models
        clf.fit(X_train_oh, y_train)
        preds_valid = clf.predict_proba(X_valid_oh)
    
    # Compute the ROC AUC score for the validation set
    roc_auc_valid = roc_auc_score(y_valid, preds_valid[:, 1])
    
    print(f"Validation ROC AUC for {name}: {roc_auc_valid:.4f}")

