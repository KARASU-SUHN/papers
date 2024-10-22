import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

# Assuming X_train, y_train, X_test are already loaded

# ----- 1. Feature Engineering based on Feature Importance -----

# Drop less important features
X_train = X_train.drop(columns=['line', 'maintenance_count'])
X_test = X_test.drop(columns=['line', 'maintenance_count'])

# Interaction terms between pressure and other important features
X_train['pressure_position_interaction'] = X_train['pressure'] * X_train['position']
X_test['pressure_position_interaction'] = X_test['pressure'] * X_test['position']

X_train['pressure_temperature_interaction'] = X_train['pressure'] * X_train[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)
X_test['pressure_temperature_interaction'] = X_test['pressure'] * X_test[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)

# Average temperature
X_train['avg_temperature'] = X_train[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)
X_test['avg_temperature'] = X_test[['temperature_1', 'temperature_2', 'temperature_3']].mean(axis=1)

# ----- 2. Model Training and Hyperparameter Tuning -----

# Split data for training and validation
X_train_full, X_valid, y_train_full, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

# Define parameter grids for XGBoost, CatBoost, and LightGBM
param_grids = {
    'XGB': {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5],
        'lambda': [1, 3, 5, 10],
        'alpha': [0, 1, 3]
    },
    'CatBoost': {
        'iterations': [500, 1000, 2000],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [3, 5, 7, 9],
        'l2_leaf_reg': [1, 3, 5, 10],
        'border_count': [128, 254, 512]
    },
    'LightGBM': {
        'n_estimators': [500, 1000, 2000],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100, 200],
        'max_depth': [-1, 10, 20, 30],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 1, 3, 5],
        'reg_lambda': [0, 1, 3, 5]
    }
}

# Initialize models
xgb_model = XGBClassifier(eval_metric='auc', use_label_encoder=False)
catboost_model = CatBoostClassifier(verbose=0)
lgbm_model = LGBMClassifier()

# StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5)

# Tune XGBoost
xgb_search = RandomizedSearchCV(xgb_model, param_distributions=param_grids['XGB'], n_iter=10, cv=skf, scoring='roc_auc', random_state=42)
xgb_search.fit(X_train_full, y_train_full)
print(f"Best XGBoost ROC AUC: {xgb_search.best_score_:.4f}")

# Tune CatBoost
catboost_search = RandomizedSearchCV(catboost_model, param_distributions=param_grids['CatBoost'], n_iter=10, cv=skf, scoring='roc_auc', random_state=42)
catboost_search.fit(X_train_full, y_train_full)
print(f"Best CatBoost ROC AUC: {catboost_search.best_score_:.4f}")

# Tune LightGBM
lgbm_search = RandomizedSearchCV(lgbm_model, param_distributions=param_grids['LightGBM'], n_iter=10, cv=skf, scoring='roc_auc', random_state=42)
lgbm_search.fit(X_train_full, y_train_full)
print(f"Best LightGBM ROC AUC: {lgbm_search.best_score_:.4f}")

# ----- 3. Ensembling with Stacking -----

# Use Logistic Regression as the final estimator
stacking_model = StackingClassifier(
    estimators=[
        ('xgb', xgb_search.best_estimator_),
        ('catboost', catboost_search.best_estimator_),
        ('lightgbm', lgbm_search.best_estimator_)
    ],
    final_estimator=LogisticRegression(),
    cv=skf
)
stacking_model.fit(X_train_full, y_train_full)
stacking_preds = stacking_model.predict_proba(X_valid)[:, 1]
stacking_roc_auc = roc_auc_score(y_valid, stacking_preds)
print(f"Stacking Model ROC AUC: {stacking_roc_auc:.4f}")

# ----- 4. Calibration -----

# Calibrate the best model
calibrated_model = CalibratedClassifierCV(xgb_search.best_estimator_, method='sigmoid')
calibrated_model.fit(X_train_full, y_train_full)
calibrated_preds = calibrated_model.predict_proba(X_valid)[:, 1]
calibrated_roc_auc = roc_auc_score(y_valid, calibrated_preds)
print(f"Calibrated XGBoost ROC AUC: {calibrated_roc_auc:.4f}")

# ----- 5. SHAP for Interpretation -----

# SHAP explanation for XGBoost
# Define SHAP explanation for the best model (XGBoost, CatBoost, or LightGBM)
best_model_name = None
best_model = None

if stacking_roc_auc >= max(calibrated_roc_auc, xgb_search.best_score_, catboost_search.best_score_, lgbm_search.best_score_):
    best_model_name = 'Stacking Model'
    best_model = stacking_model
elif calibrated_roc_auc >= max(stacking_roc_auc, xgb_search.best_score_, catboost_search.best_score_, lgbm_search.best_score_):
    best_model_name = 'Calibrated XGBoost'
    best_model = calibrated_model
elif xgb_search.best_score_ >= max(catboost_search.best_score_, lgbm_search.best_score_):
    best_model_name = 'XGBoost'
    best_model = xgb_search.best_estimator_
elif catboost_search.best_score_ >= max(xgb_search.best_score_, lgbm_search.best_score_):
    best_model_name = 'CatBoost'
    best_model = catboost_search.best_estimator_
else:
    best_model_name = 'LightGBM'
    best_model = lgbm_search.best_estimator_

# Print which model was selected as the best
print(f"The best model is: {best_model_name}")

# Create a SHAP explainer for the best model
if best_model_name in ['XGBoost', 'LightGBM']:
    explainer = shap.Explainer(best_model)
    shap_values = explainer(X_train_full)
elif best_model_name == 'CatBoost':
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train_full)

# Generate SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X_train_full, show=False)  # Use show=False to suppress immediate display
plt.title(f"SHAP Summary for {best_model_name}")
plt.savefig(f"shap_summary_{best_model_name}.png", bbox_inches='tight')
plt.close()

print(f"SHAP summary plot saved as 'shap_summary_{best_model_name}.png'")
# ----- 6. Predictions on Test Data -----

# Make predictions on the test set using the stacking model
test_preds = stacking_model.predict_proba(X_test)[:, 1]
df_submission = pd.DataFrame({"prediction": test_preds})
df_submission.to_csv("submission.csv", index=False)

print("Submission saved to 'submission.csv'")



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





# +++++++++++++++++

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


