# File: modelling.py (Retraining Script - Kriteria 3 Advanced - Final Fix)
import mlflow
import pandas as pd
import numpy as np 
import time
import os
import sys
import shutil
import dagshub 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    log_loss, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt 
from mlflow.tracking import MlflowClient 

# Opsi display
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import warnings
warnings.filterwarnings('ignore')

# --- Fungsi Training & Logging  ---
def train_and_log_model(X_train, y_train, X_test, y_test, params, model_name="RandomForestClassifier"):
    """
    Melatih RandomForest dan log manual ke MLflow.
    """
    print(f"  Training model with params: {params}", file=sys.stderr)
    
    # Log Parameter
    mlflow.log_params(params)
    
    model = RandomForestClassifier(random_state=42, **params) 

    # Latih Model & Catat Waktu
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_duration = end_time - start_time
    mlflow.log_metric("training_duration_seconds", training_duration)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Log Metrik
    mlflow.log_metric("test_accuracy_score", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred, zero_division=0))
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba))
    
    # Log Specificity
    tn, fp, fn, tp = 0,0,0,0
    try:
        cm_values = confusion_matrix(y_test, y_pred).ravel()
        if len(cm_values) == 4: tn, fp, fn, tp = cm_values
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        mlflow.log_metric("test_specificity", specificity)
    except ValueError:
        mlflow.log_metric("test_specificity", 0.0)

    # Log Model Artifact
    mlflow.sklearn.log_model(model, "model") 
    print("  Model artifact logged to MLflow.")
    
    # Simpan model ke lokal '../model' directory
    # Path relatif dari skrip modelling.py (di dalam MLProject/) ke folder model (di luar MLProject/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.abspath(os.path.join(script_dir, "../model")) # Naik satu level, masuk ke /model
    
    try:
        if os.path.exists(local_model_dir):
            shutil.rmtree(local_model_dir) # Hapus jika sudah ada
        # Gunakan save_model
        mlflow.sklearn.save_model(model, local_model_dir, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        print(f"  Model saved locally to {local_model_dir}")
    except Exception as e:
        print(f"  Warning: Failed to save model locally to {local_model_dir}. Error: {e}")

    return { # Kembalikan metrik utama jika perlu
        "model": model,
        "test_f1_score": f1_score(y_test, y_pred),
    }

# --- Automated Retraining Script ---
if __name__ == "__main__":
    
    # --- 0. Konfigurasi DagsHub & MLflow Client (WAJIB DI AWAL) ---
    DAGSHUB_USER = 'Yona1201' 
    DAGSHUB_REPO = 'msml-proyek-briliona' 
    try:
        # PENTING: Inisialisasi DagsHub di awal skrip
        dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
        print(f"MLflow Tracking URI set to DagsHub: {mlflow.get_tracking_uri()}")
    except Exception as e:
        print(f"FATAL PYTHON ERROR: DagsHub Init Failed. Error: {e}", file=sys.stderr)
        sys.exit(1) # Keluar jika init gagal

    # Setup Eksperimen
    CI_EXPERIMENT_NAME = "Diabetes_Prediction_CI_Retraining"
    mlflow.set_experiment(CI_EXPERIMENT_NAME)

    TUNING_EXPERIMENT_NAME = "Diabetes_Prediction_Hyperparameter_Tuning"
    PARENT_TUNING_RUN_NAME = "ParameterGrid_Hyperparameter_Tuning_Parent_Run"

    client = MlflowClient()

    # Mulai Run MLflow baru
    with mlflow.start_run(run_name="CI_Automated_Retrain_Run_Dynamic_Params") as ci_run:
        print("\n--- Starting CI Automated Retraining ---", file=sys.stderr)

        # 1. Load Data (PATH dan TARGET DIPERBAIKI)
        # Path relatif dari folder MLProject/ (tempat skrip ini dijalankan oleh mlflow run)
        DATA_PATH = os.path.join('namadataset_preprocessing', 'diabetes_preprocessing.csv') 
        TARGET_COL = 'Diabetes_binary' 
        try:
            data = pd.read_csv(DATA_PATH) 
            if TARGET_COL not in data.columns: raise KeyError(f"Target column '{TARGET_COL}' not found.")
            print(f"Dataset loaded from {DATA_PATH}")
        except Exception as e: 
            print(f"FATAL PYTHON ERROR: Data Loading Failed. Path={DATA_PATH}. Error: {e}", file=sys.stderr)
            sys.exit(1) # Keluar jika gagal
             
        X = data.drop(TARGET_COL, axis=1)
        y = data[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # --- Scaling ---
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        if not numerical_features: 
             print("FATAL PYTHON ERROR: No numerical features found for scaling.", file=sys.stderr)
             sys.exit(1)
        scaler = StandardScaler()
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])
        print("Features scaled.")

        # 2. Retrieve best parameters 
        best_retrain_params = {}
        try:
            # Mengambil parameter terbaik dari run tuning sebelumnya
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
            model_params_to_extract = ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'min_samples_split']
            
            # Konversi tipe data parameter
            for k in model_params_to_extract:
                 if k in raw_params:
                     v = raw_params[k]
                     try: 
                         if k in ['n_estimators', 'min_samples_leaf', 'min_samples_split']: best_retrain_params[k] = int(v)
                         elif k == 'max_depth': best_retrain_params[k] = int(v) if v not in ['None', None, ''] else None
                         elif k == 'max_features': best_retrain_params[k] = v if v in ['sqrt', 'log2'] else (float(v) if '.' in str(v) else int(v))
                         else: best_retrain_params[k] = v 
                     except (ValueError, TypeError): print(f"Warn: Skip param {k}='{v}'.")
            
            print(f"Retrieved best parameters from run {best_child_run_id}: {best_retrain_params}", file=sys.stderr)
            mlflow.log_param("retrained_from_best_run_id", best_child_run_id)

        except Exception as e:
            # Jika ada error saat mencari run, cetak error dan gunakan fallback params
            print(f"PYTHON WARNING: Error retrieving best parameters: {e}. Using defaults.", file=sys.stderr)
            # Jangan exit, gunakan default
            best_retrain_params = { 'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 2 }

        best_retrain_params.pop('random_state', None)
        mlflow.log_params(best_retrain_params)
        mlflow.log_param("retrain_data_path", DATA_PATH)

        # 3. Call training function
        train_and_log_model(X_train, y_train, X_test, y_test, best_retrain_params)
        
        # --- Cetak Final Run ID (Wajib untuk Workflow Parsing) ---
        print(f"Final Run ID: {ci_run.info.run_id}") # Cetak ID ke stdout
        print("\n--- CI Automated Retraining Complete ---", file=sys.stderr)