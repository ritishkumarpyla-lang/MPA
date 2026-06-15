import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import time
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

# 1. Load Dataset
data = pd.read_csv(r"C:\Users\HP\Downloads\phone_addiction_dataset (3) (1).csv")

# 2. Data Cleaning & Preprocessing
target_col = "Addiction_Level"
data = data.drop_duplicates()
data = data.dropna(subset=[target_col])

# Drop unique identifier or high-cardinality text columns
columns_to_drop = ["Name", "Location"] 
data = data.drop(columns=columns_to_drop, errors="ignore")

# Fill missing numerical values with median (just in case)
num_cols = data.select_dtypes(include=['number']).columns.drop(target_col)
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Fill missing categorical values with 'Unknown' and apply One-Hot Encoding
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = data[cat_cols].fillna('Unknown')
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# 3. Segregate features and Target
X = data.drop(columns=[target_col])
y = data[target_col]

# 4. Train / Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Objective Functions for Regression ---

def objective_knn(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    
    model = KNeighborsRegressor(
        n_neighbors=trial.suggest_int('n_neighbors', 3, 21, 2),
        weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
        p=trial.suggest_int('p', 1, 3)
    )
    
    pipeline = Pipeline([('Scaler', scaler), ('Model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()


def objective_dt(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    
    model = DecisionTreeRegressor(
        criterion=trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error']),
        max_depth=trial.suggest_int('max_depth', 2, 30),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
        random_state=42
    )
    
    pipeline = Pipeline([('Scaler', scaler), ('Model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()


def objective_svr(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    params = {
        'C': trial.suggest_float('C', 1e-1, 1e3, log=True),
        'kernel': kernel
    }

    if kernel in ['rbf', 'poly']:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 4)

    pipeline = Pipeline([('Scaler', scaler), ('Model', SVR(**params))])
    kf = KFold(n_splits=3, shuffle=True, random_state=42) 
    return cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()


def objective_ridge(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

    model = Ridge(alpha=trial.suggest_float('alpha', 1e-3, 1e3, log=True))
    
    pipeline = Pipeline([('Scaler', scaler), ('Model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()


def objective_rf(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
        criterion=trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse']),
        max_depth=trial.suggest_int('max_depth', 5, 40),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([('Scaler', scaler), ('Model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()


def objective_gb(trial):
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()

    model = GradientBoostingRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 2, 10),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        random_state=42
    )

    pipeline = Pipeline([('Scaler', scaler), ('Model', model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=kf).mean()


# Map model names to objective functions
objectives = {
    "Ridge": objective_ridge,
    "KNN": objective_knn,
    "DecisionTree": objective_dt,
    "SVR": objective_svr,
    "RandomForest": objective_rf,
    "GradientBoosting": objective_gb
}

# Set experiment
mlflow.set_experiment("PHONE_ADDICTION_REGRESSION")

results = {}
model_dict = {model: i for i, model in enumerate(objectives.keys())}
scaler_dict = {'standard': 0, 'minmax': 1}

# Loop through each algorithm
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    # Wrap the entire process inside an explicit active MLflow run
    with mlflow.start_run(run_name=model_name):
        
        mlflow_cb = MLflowCallback(
            tracking_uri=None,              
            metric_name="cv_r2",            
            mlflow_kwargs={"nested": True}
        )

        study = optuna.create_study(direction="maximize")

        start_fit = time.time()
        study.optimize(obj_fn, n_trials=10, callbacks=[mlflow_cb]) # Using 10 trials
        fit_time = time.time() - start_fit

        print(f"Best CV R2 Score for {model_name}: {study.best_value:.4f}")
        best_params = study.best_params
        results[model_name] = {"best_params": best_params, "best_cv_r2": study.best_value}

        # Extract best scaler
        final_scaler = StandardScaler() if best_params["scaler_type"] == "standard" else MinMaxScaler()

        # Re-build the best model for final evaluation
        if model_name == "KNN":
            final_model = KNeighborsRegressor(
                n_neighbors=best_params["n_neighbors"], weights=best_params["weights"], p=best_params["p"]
            )
        elif model_name == "DecisionTree":
            final_model = DecisionTreeRegressor(
                criterion=best_params["criterion"], max_depth=best_params["max_depth"],
                min_samples_split=best_params["min_samples_split"], min_samples_leaf=best_params["min_samples_leaf"],
                max_features=best_params["max_features"], random_state=42
            )
        elif model_name == "SVR":
            params = {"kernel": best_params["kernel"], "C": best_params["C"]}
            if best_params["kernel"] in ["rbf", "poly"]: params["gamma"] = best_params["gamma"]
            if best_params["kernel"] == "poly": params["degree"] = best_params["degree"]
            final_model = SVR(**params)
        elif model_name == "Ridge":
            final_model = Ridge(alpha=best_params["alpha"])
        elif model_name == "RandomForest":
            final_model = RandomForestRegressor(
                n_estimators=best_params["n_estimators"], criterion=best_params["criterion"],
                max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"], max_features=best_params["max_features"],
                bootstrap=best_params["bootstrap"], random_state=42, n_jobs=-1
            )
        elif model_name == "GradientBoosting":
            final_model = GradientBoostingRegressor(
                n_estimators=best_params["n_estimators"], learning_rate=best_params["learning_rate"],
                max_depth=best_params["max_depth"], min_samples_split=best_params["min_samples_split"],
                min_samples_leaf=best_params["min_samples_leaf"], max_features=best_params["max_features"],
                subsample=best_params["subsample"], random_state=42
            )

        # Build and fit the final Pipeline
        final_pipeline = Pipeline([('Scaler', final_scaler), ('Model', final_model)])
        final_pipeline.fit(X_train, y_train)

        # Evaluate
        start_test = time.time()
        y_pred = final_pipeline.predict(X_test)
        test_time = time.time() - start_test

        y_train_pred = final_pipeline.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"{model_name} Training R2: {train_r2:.4f}, Testing R2: {test_r2:.4f}, Testing RMSE: {test_rmse:.2f}")
        print(f"{model_name} Fit Time: {fit_time:.2f}s, Test Time: {test_time:.2f}s")

        # Save model
        model_path = f"{model_name}_final_model.pkl"
        joblib.dump(final_pipeline, model_path)
        model_size = os.path.getsize(model_path)
        
        # Log metrics to MLflow
        mlflow.log_metric(f"model_id", model_dict[model_name])
        mlflow.log_metric(f"Scaler_id", scaler_dict[best_params["scaler_type"]]) 
        mlflow.log_metric(f"train_r2", train_r2)
        mlflow.log_metric(f"test_r2", test_r2)
        mlflow.log_metric(f"test_rmse", test_rmse)
        mlflow.log_metric(f"train_time", fit_time)
        mlflow.log_metric(f"test_time", test_time)
        mlflow.log_metric(f"model_size", model_size)
        mlflow.sklearn.log_model(final_pipeline, artifact_path=f"{model_name}_addiction_model")
        
        os.remove(model_path)

        results[model_name].update({
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "fit_time": fit_time,
            "test_time": test_time,
            "model_size_bytes": model_size
        })

# Summary
print("\n--- Summary ---")
for model_name, res in results.items():
    print(f"{model_name}: CV R2={res['best_cv_r2']:.4f}, Train R2={res['train_r2']:.4f}, "
          f"Test R2={res['test_r2']:.4f}, Test RMSE={res['test_rmse']:.2f}, Fit Time={res['fit_time']:.2f}s, "
          f"Model Size={res['model_size_bytes']} bytes")