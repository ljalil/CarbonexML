import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

import os
import logging
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/multitask_model", exist_ok=True)


class MultiTaskNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Shared trunk: P, T -> Shared Feature S
        self.trunk = nn.Sequential(
            nn.Linear(2, 128),  # P, T input
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Fugacity head: f(P, T)
        self.fugacity_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Chemical potential head: μ(P, T)
        self.chemical_potential_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Concentration branch (C1–6)
        self.conc_branch = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Activity head: a(P, T, C1–6)
        self.activity_head = nn.Sequential(
            nn.Linear(64 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, p, t, c):
        """
        p: Tensor of shape (batch_size, 1) Pressure
        t: Tensor of shape (batch_size, 1) Temperature
        c: Tensor of shape (batch_size, 6) Concentrations of ions
        """
        x = torch.cat([p, t], dim=1)  # shape: (batch_size, 2)
        shared = self.trunk(x)  # shared representation S

        fugacity = self.fugacity_head(shared)
        chemical_potential = self.chemical_potential_head(shared)

        c_feat = self.conc_branch(c)
        activity_input = torch.cat([shared, c_feat], dim=1)
        activity = self.activity_head(activity_input)

        return fugacity, chemical_potential, activity


# --- Configuration Parameters ---
data_params_dict = {
    # Paths for DuanSun data (for Stage 1 and part of Stage 2)
    "duansun_train_path": "data/processed/DuanSun_train.csv",
    "duansun_test_path": "data/processed/DuanSun_test.csv",
    # Paths for Experimental data (for Stage 2 only)
    "experimental_train_path": "data/processed/experimental_train.csv",
    "experimental_test_path": "data/processed/experimental_test.csv",
    # Path for combined experimental data for plotting, etc.
    "all_experimental_data_path": "data/processed/experimental.csv", 
    # General feature columns (inputs to the model)
    "feature_columns": [
        "Temperature (K)",
        "Pressure (MPa)",
        "Na+", "K+", "Mg+2", "Ca+2", "SO4-2", "Cl-",
    ],
    # Targets for Stage 1 (only available in DuanSun data)
    "stage1_target_columns": ["Fugacity", "ChemicalPotential", "Activity"],
    # Target for Stage 2 (available in both DuanSun and Experimental data)
    "stage2_target_column": "Dissolved CO2 (mol/kg)",
    "scaler_path": "data/processed/feature_scaler.joblib",
}

multitask_params_dict = {
    "output_dir": "results/multitask_model",
    "stage1_epochs": 100, # Epochs for Stage 1 (F, Mu, A) training
    "stage2_epochs": 50, # Epochs for Stage 2 (CO2) fine-tuning
    "batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0.0,
    "device": "cpu", # Set to "cuda" if you have a GPU
}

# --- Utility Functions and Classes ---

def calculate_CO2_vap_mol_frac(P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mole fraction of CO2 in the vapor phase using PyTorch tensors.
    """
    # convert pressure from MPa to bar
    P_bar = P * 10.0 # P is (batch_size, 1) or (N,)

    Tc = 647.29
    Pc = 220.85

    t = (T - Tc) / Tc # T is (batch_size, 1) or (N,)

    # Ensure constants are tensors on the correct device and dtype
    Tc_t = torch.tensor(Tc, dtype=T.dtype, device=T.device)
    Pc_t = torch.tensor(Pc, dtype=P.dtype, device=P.device)

    # Use torch operations for powers and multiplication
    # torch.pow(base, exponent) handles tensor exponents correctly
    P_water = (Pc_t * T / Tc_t) * (
        1
        - 38.640844 * torch.pow(-t, 1.9)
        + 5.8948420 * t
        + 59.876516 * torch.pow(t, 2)
        + 26.654627 * torch.pow(t, 3)
        + 10.637097 * torch.pow(t, 4)
    )

    mole_fraction = (P_bar - P_water) / P_bar
    # Ensure mole_fraction is non-negative, possibly clip at a small epsilon
    return torch.max(mole_fraction, torch.tensor(1e-6, dtype=mole_fraction.dtype, device=mole_fraction.device))

def calculate_dissolved_co2(
    fugacity: torch.Tensor,
    mu: torch.Tensor, # Chemical potential
    activity: torch.Tensor,
    xgas: torch.Tensor, # Molar fraction
    pressure_MPa: torch.Tensor
) -> torch.Tensor:
    """
    Calculates dissolved CO2 based on the given thermodynamic parameters and pressure using PyTorch tensors.
    Formula: dissolved co2 = exp(log(MolarFraction * P * Fugacity) - ChemicalPotential + Activity)
    """
    # Calculate the term inside the logarithm
    term_log_input = xgas * pressure_MPa * fugacity
    
    # Apply a small positive clip to avoid log(0) or log(negative) issues
    # as these physical quantities are expected to be positive.
    log_term = torch.log(torch.max(term_log_input, torch.tensor(1e-12, dtype=term_log_input.dtype, device=term_log_input.device)))

    calculated_co2 = torch.exp(log_term - mu + activity)
    return calculated_co2


class TabularDataset(Dataset):
    """Simple Dataset for tabular data, handles single or multi-target outputs."""
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        # If targets are 1D (e.g., (N,)), convert to 2D (N,1) for consistency with NN output
        if len(self.targets.shape) == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

class CO2Dataset(Dataset):
    """Dataset for fine-tuning on CO2 target with features, and co2 target."""
    def __init__(self, features: np.ndarray, co2: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Ensure target tensor is shape (N,1), flatten any extra dims then unsqueeze
        self.co2 = torch.tensor(co2, dtype=torch.float32).flatten().unsqueeze(1)
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int):
        return self.features[idx], self.co2[idx]

def setup_logging(log_path: str):
    """Sets up logging to file and console, clearing previous handlers."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

def _load_and_scale_data(df: pd.DataFrame, feature_cols: List[str], target_cols: List[str], scaler: StandardScaler, device: str):
    """Helper to load features/targets and apply scaling."""
    X = df[feature_cols].values
    y = df[target_cols].values if target_cols else None
    
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def train_stage1_thermo_properties(model: MultiTaskNN, scaler: StandardScaler) -> str:
    """
    Trains the MultiTaskNN for Fugacity, Chemical Potential, and Activity.
    """
    model_params = multitask_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]
    device = model_params["device"]

    log_file = os.path.join(output_dir, "stage1_training.log")
    setup_logging(log_file)
    logging.info(f"--- Starting Stage 1 Training: Fugacity, Chemical Potential, Activity ---")
    logging.info(f"Output directory: {output_dir}")

    # Load DuanSun data for Stage 1
    logging.info("Loading DuanSun data for Stage 1...")
    train_df_ds = pd.read_csv(data_params["duansun_train_path"])
    test_df_ds = pd.read_csv(data_params["duansun_test_path"])

    feature_cols = data_params["feature_columns"]
    stage1_target_cols = data_params["stage1_target_columns"]

    X_train_scaled, y_train_stage1 = _load_and_scale_data(train_df_ds, feature_cols, stage1_target_cols, scaler, device)
    X_val_scaled, y_val_stage1 = _load_and_scale_data(test_df_ds, feature_cols, stage1_target_cols, scaler, device)

    train_dataset = TabularDataset(X_train_scaled, y_train_stage1)
    val_dataset = TabularDataset(X_val_scaled, y_val_stage1)

    train_loader = DataLoader(train_dataset, batch_size=model_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_params["batch_size"], shuffle=False)
    logging.info(f"Stage 1 Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"], weight_decay=model_params.get("weight_decay", 0))
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    model_save_path = os.path.join(output_dir, "best_stage1_model.pt")

    train_losses = []
    val_losses = []

    logging.info("Starting Stage 1 training loop...")
    for epoch in range(model_params["stage1_epochs"]):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            p = batch_features[:, 1:2]  # Pressure (assuming it's the second feature)
            t = batch_features[:, 0:1]  # Temperature (assuming it's the first feature)
            c = batch_features[:, 2:]   # Ion concentrations

            optimizer.zero_grad()
            fugacity_pred, chemical_potential_pred, activity_pred = model(p, t, c)

            # Calculate individual losses
            loss_f = criterion(fugacity_pred, batch_targets[:, 0:1])
            loss_mu = criterion(chemical_potential_pred, batch_targets[:, 1:2])
            loss_a = criterion(activity_pred, batch_targets[:, 2:3])
            
            total_loss = loss_f + loss_mu + loss_a # Sum of losses

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * batch_features.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
                
                p = batch_features[:, 1:2]
                t = batch_features[:, 0:1]
                c = batch_features[:, 2:]

                fugacity_pred, chemical_potential_pred, activity_pred = model(p, t, c)
                loss_f = criterion(fugacity_pred, batch_targets[:, 0:1])
                loss_mu = criterion(chemical_potential_pred, batch_targets[:, 1:2])
                loss_a = criterion(activity_pred, batch_targets[:, 2:3])
                
                total_val_loss = loss_f + loss_mu + loss_a
                val_running_loss += total_val_loss.item() * batch_features.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        logging.info(
            f"Epoch {epoch+1}/{model_params['stage1_epochs']}, Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss - model_params["early_stopping_delta"]:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            logging.info(f"Validation loss improved. Saving model to {model_save_path}")
        else:
            epochs_no_improve += 1

        if (model_params.get("use_early_stopping", False) and epochs_no_improve >= model_params["early_stopping_patience"]):
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    plt.figure(figsize=(10, 6))
    epochs_range = list(range(1, len(train_losses) + 1))
    plt.plot(epochs_range, train_losses, label="Training Loss", marker="o", linestyle="-", color="blue")
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="x", linestyle="-", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Stage 1: Training and Validation Loss (Fugacity, Chemical Potential, Activity)")
    plt.legend()
    plt.grid(True)
    if val_losses:
        min_val_loss_value = min(val_losses)
        best_epoch_idx = val_losses.index(min_val_loss_value)
        best_epoch_num = epochs_range[best_epoch_idx]
        plt.annotate(
            f"Best val loss: {min_val_loss_value:.6f} at epoch {best_epoch_num}",
            xy=(best_epoch_num, min_val_loss_value),
            xytext=(best_epoch_num + max(1, len(epochs_range) * 0.05), min_val_loss_value * 1.1),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        )
    plot_path = os.path.join(output_dir, "stage1_training_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Stage 1 training plot saved to {plot_path}")

    logging.info("Stage 1 Training finished.")
    logging.info(f"Best validation loss for Stage 1: {best_val_loss:.6f}")
    logging.info(f"Best Stage 1 model saved at: {model_save_path}")
    return model_save_path


def train_stage2_co2_solubility(model: MultiTaskNN, scaler: StandardScaler, stage1_model_path: str) -> str:
    """
    Fine-tunes the MultiTaskNN for Dissolved CO2 solubility prediction.
    """
    model_params = multitask_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]
    device = model_params["device"]

    log_file = os.path.join(output_dir, "stage2_training.log")
    setup_logging(log_file) # Re-setup logging for Stage 2
    logging.info(f"--- Starting Stage 2 Training: Dissolved CO2 Solubility ---")

    # Load best Stage 1 model state
    logging.info(f"Loading Stage 1 model from {stage1_model_path}")
    model.load_state_dict(torch.load(stage1_model_path, map_location=device))
    model.to(device)

    # Load and combine data for Stage 2
    logging.info("Loading and combining DuanSun and Experimental data for Stage 2...")
    train_df_ds = pd.read_csv(data_params["duansun_train_path"])
    test_df_ds = pd.read_csv(data_params["duansun_test_path"])
    train_df_exp = pd.read_csv(data_params["experimental_train_path"])
    test_df_exp = pd.read_csv(data_params["experimental_test_path"])

    combined_train_df = pd.concat([train_df_ds, train_df_exp], ignore_index=True)
    combined_test_df = pd.concat([test_df_ds, test_df_exp], ignore_index=True)

    feature_cols = data_params["feature_columns"]
    stage2_target_col = data_params["stage2_target_column"]

    X_train_scaled_co2, y_train_co2 = _load_and_scale_data(combined_train_df, feature_cols, [stage2_target_col], scaler, device)
    X_val_scaled_co2, y_val_co2 = _load_and_scale_data(combined_test_df, feature_cols, [stage2_target_col], scaler, device)
    
    train_dataset_co2 = CO2Dataset(X_train_scaled_co2, y_train_co2)
    val_dataset_co2 = CO2Dataset(X_val_scaled_co2, y_val_co2)

    train_loader_co2 = DataLoader(train_dataset_co2, batch_size=model_params["batch_size"], shuffle=True)
    val_loader_co2 = DataLoader(val_dataset_co2, batch_size=model_params["batch_size"], shuffle=False)
    logging.info(f"Stage 2 Training samples: {len(train_dataset_co2)}, Validation samples: {len(val_dataset_co2)}")

    optimizer = optim.Adam(model.parameters(), lr=model_params["learning_rate"], weight_decay=model_params.get("weight_decay", 0))
    criterion_co2 = nn.MSELoss()

    best_val_loss_co2 = float("inf")
    epochs_no_improve = 0
    model_save_path = os.path.join(output_dir, "final_co2_solubility_model.pt")

    train_losses_co2 = []
    val_losses_co2 = []

    logging.info("Starting Stage 2 training loop...")
    for epoch in range(model_params["stage2_epochs"]):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets_co2 in train_loader_co2:
            batch_features, batch_targets_co2 = batch_features.to(device), batch_targets_co2.to(device)

            p_scaled = batch_features[:, 1:2] # Scaled Pressure
            t_scaled = batch_features[:, 0:1] # Scaled Temperature
            c_scaled = batch_features[:, 2:]  # Scaled Ion concentrations

            # Inverse transform P and T to get actual values for xgas calculation
            # Create a dummy array for inverse transform
            dummy_features_for_inverse = torch.zeros_like(batch_features)
            dummy_features_for_inverse[:, 0:1] = t_scaled
            dummy_features_for_inverse[:, 1:2] = p_scaled
            # Convert to numpy for scaler, then back to torch
            unscaled_features = scaler.inverse_transform(dummy_features_for_inverse.cpu().numpy())
            
            p_unscaled = torch.tensor(unscaled_features[:, 1:2], dtype=torch.float32, device=device)
            t_unscaled = torch.tensor(unscaled_features[:, 0:1], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            fugacity_pred, chemical_potential_pred, activity_pred = model(p_scaled, t_scaled, c_scaled)

            # Calculate xgas using unscaled P and T
            xgas = calculate_CO2_vap_mol_frac(p_unscaled, t_unscaled)
            
            # Calculate dissolved CO2 using predictions and unscaled P and xgas
            co2_pred = calculate_dissolved_co2(fugacity_pred, chemical_potential_pred, activity_pred, xgas, p_unscaled)
            
            loss = criterion_co2(co2_pred, batch_targets_co2)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)
        epoch_loss = running_loss / len(train_loader_co2.dataset)
        train_losses_co2.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_targets_co2 in val_loader_co2:
                batch_features, batch_targets_co2 = batch_features.to(device), batch_targets_co2.to(device)

                p_scaled = batch_features[:, 1:2]
                t_scaled = batch_features[:, 0:1]
                c_scaled = batch_features[:, 2:]

                dummy_features_for_inverse = torch.zeros_like(batch_features)
                dummy_features_for_inverse[:, 0:1] = t_scaled
                dummy_features_for_inverse[:, 1:2] = p_scaled
                unscaled_features = scaler.inverse_transform(dummy_features_for_inverse.cpu().numpy())
                
                p_unscaled = torch.tensor(unscaled_features[:, 1:2], dtype=torch.float32, device=device)
                t_unscaled = torch.tensor(unscaled_features[:, 0:1], dtype=torch.float32, device=device)

                fugacity_pred, chemical_potential_pred, activity_pred = model(p_scaled, t_scaled, c_scaled)
                xgas = calculate_CO2_vap_mol_frac(p_unscaled, t_unscaled)
                co2_pred = calculate_dissolved_co2(fugacity_pred, chemical_potential_pred, activity_pred, xgas, p_unscaled)
                
                loss = criterion_co2(co2_pred, batch_targets_co2)
                val_running_loss += loss.item() * batch_features.size(0)
        epoch_val_loss = val_running_loss / len(val_loader_co2.dataset)
        val_losses_co2.append(epoch_val_loss)

        logging.info(
            f"Epoch {epoch+1}/{model_params['stage2_epochs']}, Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss_co2 - model_params["early_stopping_delta"]:
            best_val_loss_co2 = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            logging.info(f"Validation loss improved. Saving model to {model_save_path}")
        else:
            epochs_no_improve += 1

        if (model_params.get("use_early_stopping", False) and epochs_no_improve >= model_params["early_stopping_patience"]):
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    plt.figure(figsize=(10, 6))
    epochs_range = list(range(1, len(train_losses_co2) + 1))
    plt.plot(epochs_range, train_losses_co2, label="Training Loss", marker="o", linestyle="-", color="blue")
    plt.plot(epochs_range, val_losses_co2, label="Validation Loss", marker="x", linestyle="-", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Stage 2: Training and Validation Loss (Dissolved CO2)")
    plt.legend()
    plt.grid(True)
    if val_losses_co2:
        min_val_loss_value = min(val_losses_co2)
        best_epoch_idx = val_losses_co2.index(min_val_loss_value)
        best_epoch_num = epochs_range[best_epoch_idx]
        plt.annotate(
            f"Best val loss: {min_val_loss_value:.6f} at epoch {best_epoch_num}",
            xy=(best_epoch_num, min_val_loss_value),
            xytext=(best_epoch_num + max(1, len(epochs_range) * 0.05), min_val_loss_value * 1.1),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        )
    plot_path = os.path.join(output_dir, "stage2_training_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Stage 2 training plot saved to {plot_path}")

    logging.info("Stage 2 Training finished.")
    logging.info(f"Best validation loss for Stage 2: {best_val_loss_co2:.6f}")
    logging.info(f"Final Stage 2 model saved at: {model_save_path}")
    return model_save_path


def evaluate_stage1_model(model_path: str, scaler: StandardScaler):
    """Evaluates the Stage 1 model on DuanSun test data."""
    model_params = multitask_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]
    device = model_params["device"]

    logging.info(f"\n--- Evaluating Stage 1 Model from: {model_path} ---")

    test_df_ds = pd.read_csv(data_params["duansun_test_path"])
    feature_cols = data_params["feature_columns"]
    stage1_target_cols = data_params["stage1_target_columns"]

    X_test_scaled, y_test_stage1 = _load_and_scale_data(test_df_ds, feature_cols, stage1_target_cols, scaler, device)
    
    model = MultiTaskNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        p = X_tensor[:, 1:2]
        t = X_tensor[:, 0:1]
        c = X_tensor[:, 2:]
        fugacity_pred, chemical_potential_pred, activity_pred = model(p, t, c)

    predictions = {
        "Fugacity": fugacity_pred.cpu().numpy(),
        "ChemicalPotential": chemical_potential_pred.cpu().numpy(),
        "Activity": activity_pred.cpu().numpy()
    }
    actuals = {
        "Fugacity": y_test_stage1[:, 0:1],
        "ChemicalPotential": y_test_stage1[:, 1:2],
        "Activity": y_test_stage1[:, 2:3]
    }

    logging.info("Stage 1 Model Evaluation Results (DuanSun Test Data):")
    for task_name in stage1_target_cols:
        actual = actuals[task_name]
        pred = predictions[task_name]
        mae = mean_absolute_error(actual, pred)
        mse = mean_squared_error(actual, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, pred)
        logging.info(f"  {task_name}:")
        logging.info(f"\tMAE: {mae:.4f}")
        logging.info(f"\tMSE: {mse:.4f}")
        logging.info(f"\tRMSE: {rmse:.4f}")
        logging.info(f"\tR²: {r2:.4f}")

def evaluate_stage2_model(model_path: str, scaler: StandardScaler, plot: bool = True):
    """Evaluates the Stage 2 model on combined test data for Dissolved CO2."""
    model_params = multitask_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]
    device = model_params["device"]

    logging.info(f"\n--- Evaluating Stage 2 Model from: {model_path} ---")

    test_df_ds = pd.read_csv(data_params["duansun_test_path"])
    test_df_exp = pd.read_csv(data_params["experimental_test_path"])
    combined_test_df = pd.concat([test_df_ds, test_df_exp], ignore_index=True)

    feature_cols = data_params["feature_columns"]
    stage2_target_col = data_params["stage2_target_column"]

    X_test_scaled, y_test_co2 = _load_and_scale_data(combined_test_df, feature_cols, [stage2_target_col], scaler, device)
    
    model = MultiTaskNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions_co2 = []
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        p_scaled = X_tensor[:, 1:2]
        t_scaled = X_tensor[:, 0:1]
        c_scaled = X_tensor[:, 2:]

        dummy_features_for_inverse = torch.zeros_like(X_tensor)
        dummy_features_for_inverse[:, 0:1] = t_scaled
        dummy_features_for_inverse[:, 1:2] = p_scaled
        unscaled_features = scaler.inverse_transform(dummy_features_for_inverse.cpu().numpy())
        
        p_unscaled = torch.tensor(unscaled_features[:, 1:2], dtype=torch.float32, device=device)
        t_unscaled = torch.tensor(unscaled_features[:, 0:1], dtype=torch.float32, device=device)

        fugacity_pred, chemical_potential_pred, activity_pred = model(p_scaled, t_scaled, c_scaled)
        xgas = calculate_CO2_vap_mol_frac(p_unscaled, t_unscaled)
        co2_pred = calculate_dissolved_co2(fugacity_pred, chemical_potential_pred, activity_pred, xgas, p_unscaled)
        predictions_co2.extend(co2_pred.cpu().numpy().flatten())

    predictions_co2 = np.array(predictions_co2)
    actual_co2 = y_test_co2.flatten()

    mae = mean_absolute_error(actual_co2, predictions_co2)
    mse = mean_squared_error(actual_co2, predictions_co2)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_co2, predictions_co2)

    logging.info("Stage 2 Model Evaluation Results (Combined Test Data - Dissolved CO2):")
    logging.info(f"\tMAE: {mae:.4f}")
    logging.info(f"\tMSE: {mse:.4f}")
    logging.info(f"\tRMSE: {rmse:.4f}")
    logging.info(f"\tR²: {r2:.4f}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_co2, predictions_co2, alpha=0.5, label="Predictions")

        max_val = max(np.max(actual_co2), np.max(predictions_co2))
        min_val = min(np.min(actual_co2), np.min(predictions_co2))
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )

        plt.title("Stage 2: Actual vs Predicted Dissolved CO2 (mol/kg)")
        plt.xlabel(f"Actual {stage2_target_col}")
        plt.ylabel(f"Predicted {stage2_target_col}")
        plt.legend()
        plt.grid(True)

        metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        plt.text(
            0.05,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
        )

        eval_plot_path = os.path.join(output_dir, "stage2_evaluation_plot.png")
        plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Stage 2 Evaluation plot saved to {eval_plot_path}")

def evaluate_co2_pressure_response(model_path: str, scaler: StandardScaler,
                                    temperature: float, pressure_bounds: Tuple[float, float],
                                    ion_moles: Dict[str, float], num_points: int = 100):
    """
    Evaluates the model's predictions for CO2 solubility across a range of pressures
    at a fixed temperature and ion composition.
    """
    model_params = multitask_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]
    device = model_params["device"]

    logging.info(f"\n--- Evaluating CO2 solubility vs pressure at {temperature}K from: {model_path} ---")

    feature_cols = data_params["feature_columns"]
    target_col = data_params["stage2_target_column"]
    min_p, max_p = pressure_bounds
    pressures_unscaled_np = np.linspace(min_p, max_p, num_points).reshape(-1, 1)

    # Create a template for features for prediction
    feature_template_np = np.zeros((num_points, len(feature_cols)))
    feature_indices = {col: i for i, col in enumerate(feature_cols)}

    # Set fixed temperature
    if "Temperature (K)" in feature_indices:
        feature_template_np[:, feature_indices["Temperature (K)"]] = temperature
    else:
        logging.warning("Temperature (K) not in feature columns.")

    # Set varying pressure
    if "Pressure (MPa)" in feature_indices:
        feature_template_np[:, feature_indices["Pressure (MPa)"]] = pressures_unscaled_np.flatten()
    else:
        logging.warning("Pressure (MPa) not in feature columns.")

    # Set fixed ion concentrations
    for ion, value in ion_moles.items():
        if ion in feature_indices:
            feature_template_np[:, feature_indices[ion]] = value
        else:
            logging.warning(f"Ion {ion} not in feature columns.")

    # Scale the features for model input
    features_scaled_np = scaler.transform(feature_template_np)
    
    model = MultiTaskNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions_co2 = []
    with torch.no_grad():
        X_tensor = torch.tensor(features_scaled_np, dtype=torch.float32).to(device)
        
        p_scaled = X_tensor[:, 1:2]
        t_scaled = X_tensor[:, 0:1]
        c_scaled = X_tensor[:, 2:]

        # Need unscaled P and T for xgas calculation and final CO2 formula
        p_unscaled_tensor = torch.tensor(pressures_unscaled_np, dtype=torch.float32, device=device)
        t_unscaled_tensor = torch.tensor(np.full_like(pressures_unscaled_np, temperature), dtype=torch.float32, device=device)

        fugacity_pred, chemical_potential_pred, activity_pred = model(p_scaled, t_scaled, c_scaled)
        xgas = calculate_CO2_vap_mol_frac(p_unscaled_tensor, t_unscaled_tensor)
        co2_pred = calculate_dissolved_co2(fugacity_pred, chemical_potential_pred, activity_pred, xgas, p_unscaled_tensor)
        predictions_co2.extend(co2_pred.cpu().numpy().flatten())

    predictions_co2 = np.array(predictions_co2)

    plt.figure(figsize=(10, 6))
    plt.plot(
        pressures_unscaled_np.flatten(),
        predictions_co2,
        'b-',
        linewidth=2,
        label=f"Predicted CO2 Solubility"
    )
    plt.xlabel("Pressure (MPa)")
    plt.ylabel(f'Predicted {target_col}')
    plt.title(f"Predicted CO2 Solubility vs Pressure at {temperature} K")
    plt.grid(True)

    # Overlay experimental data if available
    try:
        exp_df = pd.read_csv(data_params["all_experimental_data_path"])
        mask = np.isclose(exp_df["Temperature (K)"], temperature, atol=1) # Allow slight temperature variation
        for ion, val in ion_moles.items():
            if ion in feature_indices:
                mask &= np.isclose(exp_df[ion], val, atol=0.05) # Allow slight ion variation
        exp_df_filtered = exp_df[mask]
        
        if not exp_df_filtered.empty:
            plt.scatter(
                exp_df_filtered["Pressure (MPa)"],
                exp_df_filtered[target_col],
                c="k",
                marker="x",
                label="Experimental Data",
                s=50
            )
            logging.info(f"Overlaid {len(exp_df_filtered)} experimental points.")
        else:
            logging.info("No matching experimental data found for overlay.")

    except FileNotFoundError:
        logging.warning(f"Experimental data file not found at {data_params['all_experimental_data_path']}. Cannot overlay experimental data.")
    except Exception as e:
        logging.warning(f"Could not overlay experimental data: {e}")

    plt.legend()
    filename_base = f"co2_vs_pressure_{temperature}K"
    ion_str_parts = [
        f"{ion.replace('+','p').replace('-','m').replace('2','')}{value:.1f}"
        for ion, value in ion_moles.items()
        if value > 0 and ion in feature_indices
    ]
    if ion_str_parts:
        filename_base += "_" + "_".join(ion_str_parts)

    plot_save_path = os.path.join(output_dir, f"{filename_base}.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Pressure response plot saved to {plot_save_path}")

    csv_save_path = os.path.join(output_dir, f"{filename_base}.csv")
    results_df = pd.DataFrame(
        {
            "Pressure (MPa)": pressures_unscaled_np.flatten(),
            f'Predicted {target_col}': predictions_co2,
        }
    )
    results_df.to_csv(csv_save_path, index=False)
    logging.info(f"Pressure evaluation CSV saved to {csv_save_path}")


def main_train_and_evaluate():
    """Orchestrates the two-step training and evaluation process."""
    output_dir = multitask_params_dict["output_dir"]
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Step 0: Global Scaler Preparation
    logging.info("--- Preparing Global Scaler ---")
    train_df_ds = pd.read_csv(data_params_dict["duansun_train_path"])
    train_df_exp = pd.read_csv(data_params_dict["experimental_train_path"])
    
    # Combine feature data from both training sets to fit the scaler
    combined_train_features = pd.concat([train_df_ds[data_params_dict["feature_columns"]], 
                                         train_df_exp[data_params_dict["feature_columns"]]], 
                                        ignore_index=True)
    
    scaler = StandardScaler()
    scaler.fit(combined_train_features.values)
    joblib.dump(scaler, data_params_dict["scaler_path"])
    logging.info(f"Feature scaler fitted on combined training data and saved to {data_params_dict['scaler_path']}")

    # Step 1: Train for Fugacity, Chemical Potential, Activity
    model = MultiTaskNN()
    best_stage1_model_path = train_stage1_thermo_properties(model, scaler)

    if best_stage1_model_path and os.path.exists(best_stage1_model_path):
        evaluate_stage1_model(best_stage1_model_path, scaler)
    else:
        logging.error("Stage 1 training failed or model not found. Skipping Stage 1 evaluation.")
        return # Exit if Stage 1 failed

    # Step 2: Fine-tune for Dissolved CO2
    final_co2_model_path = train_stage2_co2_solubility(model, scaler, best_stage1_model_path)

    if final_co2_model_path and os.path.exists(final_co2_model_path):
        evaluate_stage2_model(final_co2_model_path, scaler, plot=True)
        
        # Example for pressure response plot (adjust parameters as needed)
        logging.info("\n--- Evaluating CO2 solubility response to pressure ---")
        example_temperature = 304.0 # Example temperature in K
        example_pressure_bounds = (0.1, 100.0) # Example pressure range in MPa
        
        # Define base ion concentrations (e.g., pure water or typical brine)
        all_ion_keys = [
            col
            for col in data_params_dict["feature_columns"]
            if col not in ["Temperature (K)", "Pressure (MPa)"]
        ]
        # Example: pure water (0 moles for all ions)
        base_ion_moles = {ion: 0.0 for ion in all_ion_keys} 
        
        # You can add specific ion concentrations for different plots
        # e.g., base_ion_moles["Na+"] = 0.5
        # e.g., base_ion_moles["Cl-"] = 0.5

        evaluate_co2_pressure_response(
            model_path=final_co2_model_path,
            scaler=scaler,
            temperature=example_temperature,
            pressure_bounds=example_pressure_bounds,
            ion_moles=base_ion_moles,
        )
    else:
        logging.error("Stage 2 training failed or model not found. Skipping Stage 2 evaluation.")

if __name__ == "__main__":
    main_train_and_evaluate()