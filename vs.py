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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

# Step 2: Apply StandardScaler for models that need feature scaling
scaler = StandardScaler()

# Scale only for models that are sensitive to feature scaling
X_train_scaled = scaler.fit_transform(X_train_oh)
X_valid_scaled = scaler.transform(X_valid_oh)
X_test_scaled = scaler.transform(X_test_oh)

# Step 3: Create a dictionary of classifiers, including CatBoost and LightGBM
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

# Step 4: Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation results
results = {}
best_model = None
best_auc = 0

# Perform k-fold cross-validation for each classifier
for name, clf in classifiers.items():
    if name in ['CatBoost', 'LightGBM', 'Random Forest', 'Decision Tree', 'Gradient Boosting', 'XGB']:
        # Use the raw categorical data (without scaling) for tree-based models
        preds = cross_val_predict(clf, X_train, y_train, cv=kf, method="predict_proba")
    else:
        # Use the scaled data for models that require scaling
        preds = cross_val_predict(clf, X_train_scaled, y_train, cv=kf, method="predict_proba")
    
    # Compute ROC AUC score using the second column (positive class probabilities)
    roc_auc = roc_auc_score(y_train, preds[:, 1])
    
    results[name] = roc_auc
    print(f"{name}: ROC AUC = {roc_auc:.4f}")
    
    # Track the best model based on ROC AUC
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model = clf

# Step 5: Train the best model on the entire training set and predict on the test set
print(f"\nBest model: {type(best_model).__name__} with ROC AUC = {best_auc:.4f}")

# Train the best model
if type(best_model).__name__ in ['CatBoostClassifier', 'LGBMClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'GradientBoostingClassifier', 'XGBClassifier']:
    # Use the raw data for tree-based models
    best_model.fit(X_train, y_train)
    preds_test = best_model.predict_proba(X_test)[:, 1]
else:
    # Use the scaled data for models that require scaling
    best_model.fit(X_train_scaled, y_train)
    preds_test = best_model.predict_proba(X_test_scaled)[:, 1]

# Step 6: Merge the predictions and save as a CSV
df_submission = pd.DataFrame()
df_submission["prediction"] = preds_test

# Save to CSV
df_submission.to_csv("submission.csv", index=False)

print("\nSubmission file 'submission.csv' has been saved.")

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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Assuming X_train, y_train, X_valid, y_valid, X_test are already defined

# Categorical columns
categorical_columns = ["line", "tray_no", "position"]

# Initialize the OneHotEncoder for non-tree-based models
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Fit and transform the categorical columns for non-tree-based models
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_valid_encoded = encoder.transform(X_valid[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Combine the encoded categorical data with the continuous features (including maintenance_count)
X_train_combined = np.hstack((X_train.drop(columns=categorical_columns).values, X_train_encoded))
X_valid_combined = np.hstack((X_valid.drop(columns=categorical_columns).values, X_valid_encoded))
X_test_combined = np.hstack((X_test.drop(columns=categorical_columns).values, X_test_encoded))

# Scale the data for non-tree-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_valid_scaled = scaler.transform(X_valid_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Use tree-based models directly with raw data (including maintenance_count as is)
X_train_tree_based = X_train.drop(columns=categorical_columns).values  # raw integer data for tree-based models
X_valid_tree_based = X_valid.drop(columns=categorical_columns).values
X_test_tree_based = X_test.drop(columns=categorical_columns).values

# Step 3: Create a dictionary of models and their parameter grids for RandomizedSearchCV
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'K Neighbors': KNeighborsClassifier(),
    'SVC': SVC(probability=True),  # Enable probability output for SVC
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=1000),
    'XGB': XGBClassifier(eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(verbose=0),
    'LightGBM': LGBMClassifier()
}

# Parameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'K Neighbors': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [3, 5, 7]
    },
    'MLP': {
        'hidden_layer_sizes': [(50, 50), (100,), (100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01]
    },
    'XGB': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'CatBoost': {
        'iterations': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'depth': [3, 5, 7],
        'l2_leaf_reg': [1, 3, 5]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'min_child_samples': [20, 50, 100]
    }
}

# Step 4: Define a function to perform RandomizedSearchCV for each model and save results
def tune_model(clf, param_grid, X_train, y_train, model_name, result_file):
    search = RandomizedSearchCV(clf, param_grid, n_iter=10, cv=5, scoring='roc_auc', verbose=1, random_state=42)
    search.fit(X_train, y_train)
    
    # Write the results to the text file
    with open(result_file, 'a') as f:
        f.write(f"\nModel: {model_name}\n")
        f.write(f"Best parameters: {search.best_params_}\n")
        f.write(f"Best cross-validation ROC AUC score: {search.best_score_:.4f}\n\n")
    
    return search.best_estimator_

# Step 5: Tune each model using RandomizedSearchCV and save the best parameters
result_file = "model_results.txt"
best_estimators = {}
for name, clf in classifiers.items():
    print(f"Tuning {name}...")
    if name in ['Logistic Regression', 'K Neighbors', 'SVC', 'MLP']:
        # Use scaled data for non-tree-based models
        best_estimators[name] = tune_model(clf, param_grids[name], X_train_scaled, y_train, name, result_file)
    else:
        # Use raw integer data for tree-based models
        best_estimators[name] = tune_model(clf, param_grids[name], X_train_tree_based, y_train, name, result_file)

# Step 6: Evaluate each model on the validation set and save the results to the text file
best_model_name = None
best_roc_auc = 0

with open(result_file, 'a') as f:
    f.write("Validation Results:\n")
    
    for name, model in best_estimators.items():
        if name in ['Logistic Regression', 'K Neighbors', 'SVC', 'MLP']:
            preds_valid = model.predict_proba(X_valid_scaled)[:, 1]
        else:
            preds_valid = model.predict_proba(X_valid_tree_based)[:, 1]
        
        roc_auc_valid = roc_auc_score(y_valid, preds_valid)
        f.write(f"{name} - Validation ROC AUC: {roc_auc_valid:.4f}\n")
        
        # Keep track of the best model
        if roc_auc_valid > best_roc_auc:
            best_roc_auc = roc_auc_valid
            best_model_name = name

# Step 7: Save final test predictions from the best model
best_model = best_estimators[best_model_name]
if best_model_name in ['Logistic Regression', 'K Neighbors', 'SVC', 'MLP']:
    preds_test = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    preds_test = best_model.predict_proba(X_test_tree_based)[:, 1]

df_submission = pd.DataFrame({"prediction": preds_test})
df_submission.to_csv("submission.csv", index=False)

# Log the best model in the result file
with open(result_file, 'a') as f:
    f.write(f"\nBest model selected: {best_model_name} with ROC AUC: {best_roc_auc:.4f}\n")

print("Submission saved to 'submission.csv'")
print(f"Best model: {best_model_name} with ROC AUC: {best_roc_auc:.4f}")
print("Model results saved to 'model_results.txt'")



import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Assuming X_train, y_train, X_valid, y_valid, X_test are already defined

# Categorical columns
categorical_columns = ["line", "tray_no", "position"]

# Initialize the OneHotEncoder for non-tree-based models
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Fit and transform the categorical columns for non-tree-based models
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_valid_encoded = encoder.transform(X_valid[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Combine the encoded categorical data with the continuous features (including maintenance_count)
X_train_combined = np.hstack((X_train.drop(columns=categorical_columns).values, X_train_encoded))
X_valid_combined = np.hstack((X_valid.drop(columns=categorical_columns).values, X_valid_encoded))
X_test_combined = np.hstack((X_test.drop(columns=categorical_columns).values, X_test_encoded))

# Scale the data for non-tree-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_combined)
X_valid_scaled = scaler.transform(X_valid_combined)
X_test_scaled = scaler.transform(X_test_combined)

# Use tree-based models directly with raw data (including maintenance_count as is)
X_train_tree_based = X_train.drop(columns=categorical_columns).values  # raw integer data for tree-based models
X_valid_tree_based = X_valid.drop(columns=categorical_columns).values
X_test_tree_based = X_test.drop(columns=categorical_columns).values

# ----- Feature Engineering based on Feature Importance -----

# 1. Remove less important features 'line' and 'maintenance_count'
X_train = X_train.drop(columns=['line', 'maintenance_count'])
X_valid = X_valid.drop(columns=['line', 'maintenance_count'])
X_test = X_test.drop(columns=['line', 'maintenance_count'])

# 2. Create interaction terms based on important features

# Pressure and Position Interaction
X_train['pressure_position_interaction'] = X_train['pressure'] * X_train['position']
X_valid['pressure_position_interaction'] = X_valid['pressure'] * X_valid['position']
X_test['pressure_position_interaction'] = X_test['pressure'] * X_test['position']

# Pressure and Temperature Interaction
X_train['pressure_temperature_interaction'] = X_train['pressure'] * X_train[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)
X_valid['pressure_temperature_interaction'] = X_valid['pressure'] * X_valid[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)
X_test['pressure_temperature_interaction'] = X_test['pressure'] * X_test[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)

# 3. Average Temperature Feature
X_train['avg_temperature'] = X_train[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)
X_valid['avg_temperature'] = X_valid[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)
X_test['avg_temperature'] = X_test[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)

# Step 3: Create a dictionary of models and their parameter grids for RandomizedSearchCV
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'K Neighbors': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'MLP': MLPClassifier(max_iter=1000),
    'XGB': XGBClassifier(eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(verbose=0),
    'LightGBM': LGBMClassifier()
}

# Parameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'K Neighbors': {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'Random Forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [3, 5, 7]
    },
    'MLP': {
        'hidden_layer_sizes': [(50, 50), (100,), (100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01]
    },
    'XGB': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    },
    'CatBoost': {
        'iterations': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'depth': [3, 5, 7],
        'l2_leaf_reg': [1, 3, 5]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.05],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20],
        'min_child_samples': [20, 50, 100]
    }
}

# Step 4: Define a function to perform RandomizedSearchCV for each model and save results
def tune_model(clf, param_grid, X_train, y_train, model_name, result_file):
    search = RandomizedSearchCV(clf, param_grid, n_iter=10, cv=5, scoring='roc_auc', verbose=1, random_state=42)
    search.fit(X_train, y_train)
    
    # Write the results to the text file
    with open(result_file, 'a') as f:
        f.write(f"\nModel: {model_name}\n")
        f.write(f"Best parameters: {search.best_params_}\n")
        f.write(f"Best cross-validation ROC AUC score: {search.best_score_:.4f}\n\n")
    
    return search.best_estimator_

# Step 5: Tune each model using RandomizedSearchCV and save the best parameters
result_file = "model_results.txt"
best_estimators = {}
for name, clf in classifiers.items():
    print(f"Tuning {name}...")
    if name in ['Logistic Regression', 'K Neighbors', 'MLP']:
        # Use scaled data for non-tree-based models
        best_estimators[name] = tune_model(clf, param_grids[name], X_train_scaled, y_train, name, result_file)
    else:
        # Use raw integer data for tree-based models
        best_estimators[name] = tune_model(clf, param_grids[name], X_train_tree_based, y_train, name, result_file)

# Step 6: Evaluate each model on the validation set and save the results to the text file
best_model_name = None
best_roc_auc = 0

with open(result_file, 'a') as f:
    f.write("Validation Results:\n")
    
    for name, model in best_estimators.items():
        if name in ['Logistic Regression', 'K Neighbors', 'MLP']:
            preds_valid = model.predict_proba(X_valid_scaled)[:, 1]
        else:
            preds_valid = model.predict


