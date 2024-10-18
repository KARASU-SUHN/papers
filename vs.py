import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from xgboost import XGBClassifier

# Assuming X_train, y_train, X_valid, y_valid, X_test are already defined

# Create a dictionary of classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'K Neighbors': KNeighborsClassifier(),
    'SVC': SVC(),
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
    scores = cross_val_score(clf, X_train, y_train, cv=kf, scoring='accuracy')
    results[name] = scores
    print(f"{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
