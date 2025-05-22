import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import logging
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
data_params_dict = {
    "simulation_data_path": "data/processed/simulation.csv",
    "experimental_train_path": "data/processed/experimental_train.csv",
    "experimental_test_path": "data/processed/experimental_test.csv",
    "feature_columns": [
        "Temperature (K)",
        "Pressure (MPa)",
        #"T_critical_distance",
        #"P_critical_distance",
        "Na+",
        "K+",
        "Mg+2",
        "Ca+2",
        "SO4-2",
        "Cl-",
    ],
    "target_column": "Dissolved CO2 (mol/kg)",
    "phreeqc_database_folder": "/usr/local/share/doc/phreeqc/database/",
    "temp_files_path": "temp",
    "scaler_path": "data/processed/feature_scaler.joblib",
}

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_unified_scaler():
    train_data_path = data_params_dict['experimental_train_path']
    feature_cols = data_params_dict['feature_columns']
    scaler_path = data_params_dict['scaler_path']
    
    # Create directory for scaler if it doesn't exist
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {train_data_path}")
    train_df = pd.read_csv(train_data_path)
    
    # Extract features for fitting the scaler
    X_train = train_df[feature_cols].values
    
    # Create and fit the scaler
    print("Creating and fitting StandardScaler on training data...")
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Save the scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    

if __name__ == "__main__":
    # Can be run as standalone script
    create_unified_scaler()