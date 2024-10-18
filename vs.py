
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score

# Assuming X_train, y_train, X_valid, y_valid, X_test are already defined

# Create a dictionary of classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'K Neighbors': KNeighborsClassifier(),
    'SVC': SVC(probability=True),  # Enable probability output for SVC
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=1000),
    'XGB': XGBClassifier(eval_metric='mlogloss')
}

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation results
results = {}

# Perform k-fold cross-validation for each classifier
for name, clf in classifiers.items():
    # Predict probabilities using cross_val_predict with method="predict_proba"
    preds = cross_val_predict(clf, X_train, y_train, cv=kf, method="predict_proba")
    
    # Compute ROC AUC score using the second column (positive class probabilities)
    roc_auc = roc_auc_score(y_train, preds[:, 1])
    
    results[name] = roc_auc
    print(f"{name}: ROC AUC = {roc_auc:.4f}")

# Train the best model and validate on the validation set (X_valid, y_valid)
for name, clf in classifiers.items():
    # Fit the classifier
    clf.fit(X_train, y_train)
    
    # Predict probabilities for the validation set
    preds_valid = clf.predict_proba(X_valid)
    
    # Compute the ROC AUC score for the validation set
    roc_auc_valid = roc_auc_score(y_valid, preds_valid[:, 1])
    
    print(f"Validation ROC AUC for {name}: {roc_auc_valid:.4f}")
