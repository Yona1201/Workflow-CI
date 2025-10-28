import numpy as np
import mlflow
import pandas as pd
import time
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
import dagshub 

# Opsi display
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Abaikan warnings
import warnings
warnings.filterwarnings('ignore')

# --- Fungsi Training & Logging (Pakai Autolog) ---
def train_and_log_model_autolog(X_train, y_train, X_test, y_test, params):
    """
    Melatih RandomForest dengan autolog MLflow.
    """
    print(f"  Training retrain model with params: {params}", file=sys.stderr)
    mlflow.sklearn.autolog(log_models=True, disable=False)

    model = RandomForestClassifier(random_state=42, **params)

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"  Retraining took {training_duration:.2f} seconds.", file=sys.stderr)
    mlflow.log_metric("retraining_duration_seconds", training_duration) 

    y_pred = model.predict(X_test)

    # Log specificity manual
    tn, fp, fn, tp = 0,0,0,0
    try:
        cm_values = confusion_matrix(y_test, y_pred).ravel()
        if len(cm_values) == 4: tn, fp, fn, tp = cm_values
        elif len(cm_values) == 1:
             if np.unique(y_test)[0] == 0 and np.unique(y_pred)[0] == 0: tn = cm_values[0]
             elif np.unique(y_test)[0] == 1 and np.unique(y_pred)[0] == 1: tp = cm_values[0]
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        print(f"  Test Specificity: {specificity:.4f}", file=sys.stderr)
        mlflow.log_metric("test_specificity", specificity)
    except ValueError:
        print("  Warning: Could not calculate specificity.")
        mlflow.log_metric("test_specificity", 0.0)


    # Log CM manual
    try:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (Retrained Test Set)")
        cm_path = "confusion_matrix_retrained.png"
        plt.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path): os.remove(cm_path)
    except Exception as e: print(f" Gagal menyimpan CM Retrained: {e}")

    # Nonaktifkan autolog setelah selesai
    mlflow.sklearn.autolog(disable=True) 
    return model

# --- Automated Retraining Script ---
if __name__ == "__main__":
    
    # --- Konfigurasi DagsHub & MLflow ---
    DAGSHUB_USER = 'Yona1201' 
    DAGSHUB_REPO = 'msml-proyek-briliona' 
    try:
        dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
        print(f"MLflow Tracking URI set to DagsHub: {mlflow.get_tracking_uri()}")
    except Exception as e:
        print(f"Error initializing DagsHub: {e}")
        sys.exit(1)

    CI_EXPERIMENT_NAME = "Diabetes_Prediction_CI_Retraining"
    mlflow.set_experiment(CI_EXPERIMENT_NAME)

    TUNING_EXPERIMENT_NAME = "Diabetes_Prediction_Hyperparameter_Tuning"
    PARENT_TUNING_RUN_NAME = "ParameterGrid_Hyperparameter_Tuning_Parent_Run"

    client = mlflow.tracking.MlflowClient()

    with mlflow.start_run(run_name="CI_Automated_Retrain_Run_Dynamic_Params") as ci_run:
        print("\n--- Starting CI Automated Retraining ---", file=sys.stderr)

        # 1. Load Data
        DATA_PATH = os.path.join('diabetes_preprocessing', 'diabetes_preprocessing.csv') 
        TARGET_COL = 'Diabetes_binary' 
        try:
            data = pd.read_csv(DATA_PATH)
            if TARGET_COL not in data.columns: raise KeyError(f"Target '{TARGET_COL}' not found.")
            print(f"Dataset loaded from {DATA_PATH}")
        except Exception as e: print(f"Error loading data: {e}"); sys.exit(1)
             
        X = data.drop(TARGET_COL, axis=1)
        y = data[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # --- Scaling ---
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        if not numerical_features: print("ERROR: No numerical features found."); sys.exit(1)
        scaler = StandardScaler()
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
        print("Features scaled.")

        # 2. Retrieve best parameters 
        best_retrain_params = {}
        try:
            experiment = client.get_experiment_by_name(TUNING_EXPERIMENT_NAME)
            if not experiment: raise ValueError(f"Experiment '{TUNING_EXPERIMENT_NAME}' not found")

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.'mlflow.runName' = '{PARENT_TUNING_RUN_NAME}' AND attributes.status = 'FINISHED'",
                order_by=["attributes.start_time DESC"], max_results=1
            )
            if not runs: raise ValueError(f"Parent run '{PARENT_TUNING_RUN_NAME}' not found")
            
            parent_run = runs[0]
            best_child_run_id = parent_run.data.params.get("best_model_run_id")
            if not best_child_run_id: raise ValueError("'best_model_run_id' not found in parent run")

            best_child_run = client.get_run(best_child_run_id)
            raw_params = best_child_run.data.params
            model_params_to_extract = [ 
                'n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'min_samples_split'
            ]
            
            best_retrain_params = {}
            for k in model_params_to_extract:
                 if k in raw_params:
                     v = raw_params[k]
                     try: # Konversi tipe data
                         if k in ['n_estimators', 'min_samples_leaf', 'min_samples_split']: 
                             best_retrain_params[k] = int(v)
                         elif k == 'max_depth': 
                             best_retrain_params[k] = int(v) if v not in ['None', None, ''] else None
                         # Handle max_features being int/float/str
                         elif k == 'max_features':
                             if v in ['sqrt', 'log2']: best_retrain_params[k] = v
                             else: best_retrain_params[k] = float(v) if '.' in str(v) else int(v) 
                         else: 
                             best_retrain_params[k] = v 
                     except (ValueError, TypeError): print(f"Warn: Skip param {k}='{v}'.")
            
            print(f"Retrieved best parameters from run {best_child_run_id}: {best_retrain_params}", file=sys.stderr)
            mlflow.log_param("retrained_from_best_run_id", best_child_run_id)

        except Exception as e:
            print(f"Error retrieving best parameters: {e}. Using defaults.", file=sys.stderr)
            best_retrain_params = { # Fallback
                'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1,
                'max_features': 'sqrt', 'min_samples_split': 2
            }

        best_retrain_params.pop('random_state', None)
        mlflow.log_params(best_retrain_params)
        mlflow.log_param("retrain_data_path", DATA_PATH)

        # 3. Call training function (versi autolog)
        trained_model = train_and_log_model_autolog(X_train, y_train, X_test, y_test, best_retrain_params)
        
        print(f"Final Run ID: {ci_run.info.run_id}")
        print("\n--- CI Automated Retraining Complete ---", file=sys.stderr)