import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler # For feature scaling

import os
import logging
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# MultiTaskNN Model Definition (provided by user)
class MultiTaskNN(nn.Module):
    """
    Multi-task neural network with a shared trunk and separate heads for each target:
    1. fugacity
    2. chemical potential (mu)
    3. activity
    4. gas-phase mole fraction (xgas)
    """
    def __init__(
        self,
        input_dim: int,
        trunk_hidden_dims: List[int],
        head_hidden_dims: List[int],
        tasks: List[str] = None
    ):
        super(MultiTaskNN, self).__init__()
        # Default task names, ensure they match the order of target columns as loaded from data
        # and are used consistently when forming outputs in forward().
        self.tasks = tasks or ["fugacity", "mu", "activity", "xgas"]

        # Build shared trunk
        dims = [input_dim] + trunk_hidden_dims
        trunk_layers = []
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i+1]))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)

        # Build one head per task
        self.heads = nn.ModuleDict()
        for task in self.tasks:
            layers = []
            in_dim = trunk_hidden_dims[-1] # Input to head is output of trunk
            for hdim in head_hidden_dims:
                layers.append(nn.Linear(in_dim, hdim))
                layers.append(nn.ReLU())
                in_dim = hdim
            # final output = 1 for each task
            layers.append(nn.Linear(in_dim, 1))
            self.heads[task] = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through shared trunk
        shared = self.trunk(x)
        # Collect each head's output
        outputs = []
        for task in self.tasks:
            out = self.heads[task](shared)
            outputs.append(out)
        # Concatenate to shape (batch_size, n_tasks)
        return torch.cat(outputs, dim=1)

# --- Configuration Parameters ---
# Data parameters including new input and multiple output columns
data_params_dict = {
    "experimental_data_path": "data/processed/experimental.csv", # Used for pressure response plots
    "experimental_train_path": "data/processed/DuanSun_train.csv",
    "experimental_test_path": "data/processed/DuanSun_test.csv",
    "feature_columns": [
        "Temperature (K)",
        "Pressure (MPa)",
        "Na+",
        "K+",
        "Mg+2",
        "Ca+2",
        "SO4-2",
        "Cl-",
    ],
    # Direct target columns that the model will predict
    "direct_target_columns": ["Fugacity", "ChemicalPotential", "Activity", "MolarFraction"],
    # Indirect target column, calculated from direct targets for evaluation only
    "indirect_target_column_for_eval": "Dissolved CO2 (mol/kg)",
    "scaler_path": "data/processed/feature_scaler.joblib", # Scaler for input features
}

# Mapping between data column names and model task names
# This ensures consistency between data loading and model's internal task handling.
# The order here should match the order of direct_target_columns in data_params_dict
TASK_NAME_MAP = {
    "Fugacity": "fugacity",
    "ChemicalPotential": "mu", # Maps 'ChemicalPotential' from data to 'mu' in model
    "Activity": "activity",
    "MolarFraction": "xgas" # Maps 'MolarFraction' from data to 'xgas' in model
}
INVERSE_TASK_NAME_MAP = {v: k for k, v in TASK_NAME_MAP.items()} # For plotting and logging (model_name -> data_name)

# Model and training parameters for the MultiTaskNN
multitask_params_dict = {
    "trunk_hidden_dims": [128, 64],  # Dimensions for the shared trunk
    "head_hidden_dims": [32],       # Dimensions for each task-specific head
    "output_dir": "results/multitask_model",
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "use_early_stopping": True,
    "early_stopping_patience": 20,
    "early_stopping_delta": 0.0,
    "device": "cpu",
    # Smoothness parameters are intentionally omitted as per your request.
}

# --- Utility Functions and Classes ---

def calculate_dissolved_co2(
    fugacity: np.ndarray,
    mu: np.ndarray, # Chemical potential
    activity: np.ndarray,
    xgas: np.ndarray, # Molar fraction
    pressure_MPa: np.ndarray
) -> np.ndarray:
    """
    Calculates dissolved CO2 based on the given thermodynamic parameters and pressure.
    Formula: dissolved co2 = exp(log(MolarFraction * P * Fugacity) - ChemicalPotential + Activity)
    
    Args:
        fugacity (np.ndarray): Predicted fugacity values.
        mu (np.ndarray): Predicted chemical potential (mu) values.
        activity (np.ndarray): Predicted activity values.
        xgas (np.ndarray): Predicted gas-phase mole fraction (xgas) values.
        pressure_MPa (np.ndarray): Actual pressure values from input data.
    
    Returns:
        np.ndarray: Calculated dissolved CO2 values.
    """
    # Ensure inputs are numpy arrays for element-wise operations
    fugacity = np.asarray(fugacity)
    mu = np.asarray(mu)
    activity = np.asarray(activity)
    xgas = np.asarray(xgas)
    pressure_MPa = np.asarray(pressure_MPa)

    # Calculate the term inside the logarithm
    term_log_input = xgas * pressure_MPa * fugacity
    
    # Apply a small positive clip to avoid log(0) or log(negative) issues
    # as these physical quantities are expected to be positive.
    log_term = np.log(np.maximum(term_log_input, 1e-12)) # 1e-12 as a small epsilon

    calculated_co2 = np.exp(log_term - mu + activity)
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

def setup_logging(log_path: str):
    """Sets up logging to file and console, clearing previous handlers."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

def train_multitask_model() -> str:
    """
    Trains the MultiTaskNN model following the specified configuration.
    """
    model_params = multitask_params_dict
    data_params = data_params_dict

    output_dir = model_params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    setup_logging(log_file)
    logging.info(f"Starting MultiTask NN training...")
    logging.info(f"Output directory: {output_dir}")

    logging.info("Loading data...")
    train_df = pd.read_csv(data_params["experimental_train_path"])
    test_df = pd.read_csv(data_params["experimental_test_path"])

    feature_cols = data_params["feature_columns"]
    direct_target_cols = data_params["direct_target_columns"] # List of direct target column names

    X_train = train_df[feature_cols].values
    y_train = train_df[direct_target_cols].values # Load only direct target columns
    X_val = test_df[feature_cols].values
    y_val = test_df[direct_target_cols].values # Load only direct target columns

    # Load or create and save StandardScaler for input features
    scaler_path = data_params["scaler_path"]
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded feature scaler from {scaler_path}")
    else:
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Created and saved new feature scaler to {scaler_path}")

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_dataset = TabularDataset(X_train_scaled, y_train)
    val_dataset = TabularDataset(X_val_scaled, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=model_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=model_params["batch_size"], shuffle=False
    )
    logging.info(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    input_dim = len(feature_cols)
    # Get model-compatible task names in the correct order
    model_task_names = [TASK_NAME_MAP[col] for col in direct_target_cols]
    model = MultiTaskNN(
        input_dim=input_dim,
        trunk_hidden_dims=model_params["trunk_hidden_dims"],
        head_hidden_dims=model_params["head_hidden_dims"],
        tasks=model_task_names # Pass task names to the model for head creation
    )

    criterion = nn.MSELoss() # MSE loss calculated over all tasks combined
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_params["learning_rate"],
        weight_decay=model_params.get("weight_decay", 0),
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = model_params.get("early_stopping_patience", 10)
    model_save_path = os.path.join(output_dir, "best_multitask_model.pt")

    train_losses = []
    val_losses = []

    logging.info("Starting training loop...")
    for epoch in range(model_params["epochs"]):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets) # Loss calculated across all tasks
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        logging.info(
            f"Epoch {epoch+1}/{model_params['epochs']}, Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            logging.info(f"Validation loss improved. Saving model to {model_save_path}")
        else:
            epochs_no_improve += 1

        if (
            model_params.get("use_early_stopping", False)
            and epochs_no_improve >= patience
        ):
            logging.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # Plotting training and validation loss
    plt.figure(figsize=(10, 6))
    epochs_range = list(range(1, len(train_losses) + 1))
    plt.plot(
        epochs_range,
        train_losses,
        label="Training Loss",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        epochs_range,
        val_losses,
        label="Validation Loss",
        marker="x",
        linestyle="-",
        color="red",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("MultiTask NN: Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    if val_losses:
        min_val_loss_value = min(val_losses)
        best_epoch_idx = val_losses.index(min_val_loss_value)
        best_epoch_num = epochs_range[best_epoch_idx]
        plt.annotate(
            f"Best val loss: {min_val_loss_value:.6f} at epoch {best_epoch_num}",
            xy=(best_epoch_num, min_val_loss_value),
            xytext=(
                best_epoch_num + max(1, len(epochs_range) * 0.05),
                min_val_loss_value * 1.1 if min_val_loss_value > 0 else 0.1,
            ),
            arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
        )
    plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Training plot saved to {plot_path}")

    logging.info("Training finished.")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")
    logging.info(f"Best model saved at: {model_save_path}")
    return model_save_path

def evaluate_trained_multitask_model(model_path: str, plot: bool = False):
    """
    Evaluates a trained MultiTaskNN model on the test dataset for each direct task
    and for the calculated Dissolved CO2.
    """
    model_params = multitask_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]

    logging.info(f"Evaluating MultiTask model from: {model_path}")

    try:
        test_df = pd.read_csv(data_params["experimental_test_path"])
    except FileNotFoundError:
        logging.error(
            f"Test data file not found: {data_params['experimental_test_path']}"
        )
        return

    feature_cols = data_params["feature_columns"]
    direct_target_cols = data_params["direct_target_columns"] # List of direct target column names
    indirect_target_col_for_eval = data_params["indirect_target_column_for_eval"]

    # X_test_df stores original feature values, useful for retrieving Pressure
    X_test_df = test_df[feature_cols]
    y_test_direct = test_df[direct_target_cols].values # Actual direct targets
    y_test_indirect_co2 = test_df[indirect_target_col_for_eval].values # Actual indirect target for evaluation

    scaler_path = data_params["scaler_path"]
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test_scaled = scaler.transform(X_test_df.values) # Scale only features
        logging.info(f"Loaded scaler from {scaler_path} for evaluation.")
    else:
        logging.warning(
            f"Scaler not found at {scaler_path} for evaluation. Using unscaled data. "
            "This might lead to incorrect results if the model was trained with scaled data."
        )
        X_test_scaled = X_test_df.values

    input_dim = len(feature_cols)
    model_task_names = [TASK_NAME_MAP[col] for col in direct_target_cols]
    model = MultiTaskNN(
        input_dim=input_dim,
        trunk_hidden_dims=model_params["trunk_hidden_dims"],
        head_hidden_dims=model_params["head_hidden_dims"],
        tasks=model_task_names
    )

    try:
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set model to evaluation mode
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Cannot evaluate.")
        return
    except Exception as e:
        logging.error(f"Error loading model state_dict from {model_path}: {e}")
        return

    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        predictions_direct = model(X_tensor).cpu().numpy() # Shape (N, n_direct_tasks)

    logging.info("MultiTask Model Evaluation Results per Direct Task:")
    metrics_by_task = {}
    for i, task_data_name in enumerate(direct_target_cols):
        y_true_task = y_test_direct[:, i]
        y_pred_task = predictions_direct[:, i]

        mae = mean_absolute_error(y_true_task, y_pred_task)
        mse = mean_squared_error(y_true_task, y_pred_task)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_task, y_pred_task)

        metrics_by_task[task_data_name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
        logging.info(f"\tTask: {task_data_name}")
        logging.info(f"\t\tMAE: {mae:.4f}")
        logging.info(f"\t\tMSE: {mse:.4f}")
        logging.info(f"\t\tRMSE: {rmse:.4f}")
        logging.info(f"\t\tR²: {r2:.4f}")

    # --- Evaluate Dissolved CO2 indirectly ---
    # Extract actual pressure from the feature_columns of the test set
    if "Pressure (MPa)" not in feature_cols:
        logging.error("Pressure (MPa) is not in feature_columns. Cannot calculate Dissolved CO2.")
        return metrics_by_task # Exit if pressure is missing

    actual_pressures = X_test_df["Pressure (MPa)"].values # Original pressure values

    # Extract predicted direct outputs for calculation, mapping from model output order to function arguments
    predicted_fugacity = predictions_direct[:, direct_target_cols.index("Fugacity")]
    predicted_mu = predictions_direct[:, direct_target_cols.index("ChemicalPotential")]
    predicted_activity = predictions_direct[:, direct_target_cols.index("Activity")]
    predicted_xgas = predictions_direct[:, direct_target_cols.index("MolarFraction")]

    calculated_dissolved_co2 = calculate_dissolved_co2(
        fugacity=predicted_fugacity,
        mu=predicted_mu,
        activity=predicted_activity,
        xgas=predicted_xgas,
        pressure_MPa=actual_pressures
    )

    # Evaluate the calculated dissolved CO2
    mae_co2 = mean_absolute_error(y_test_indirect_co2, calculated_dissolved_co2)
    mse_co2 = mean_squared_error(y_test_indirect_co2, calculated_dissolved_co2)
    rmse_co2 = np.sqrt(mse_co2)
    r2_co2 = r2_score(y_test_indirect_co2, calculated_dissolved_co2)

    logging.info("\n--- Calculated Dissolved CO2 Evaluation Results ---")
    logging.info(f"\tMAE ({indirect_target_col_for_eval}): {mae_co2:.4f}")
    logging.info(f"\tMSE ({indirect_target_col_for_eval}): {mse_co2:.4f}")
    logging.info(f"\tRMSE ({indirect_target_col_for_eval}): {rmse_co2:.4f}")
    logging.info(f"\tR² ({indirect_target_col_for_eval}): {r2_co2:.4f}")

    metrics_by_task[indirect_target_col_for_eval] = {"MAE": mae_co2, "MSE": mse_co2, "RMSE": rmse_co2, "R2": r2_co2}

    if plot:
        # Create a grid of subplots for each direct task + 1 for calculated Dissolved CO2
        num_direct_tasks = len(direct_target_cols)
        total_plots = num_direct_tasks + 1
        n_cols = 2 # Or adjust as needed
        n_rows = (total_plots + n_cols - 1) // n_cols # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        axes = axes.flatten() # Flatten array for easy iteration, even if 1x1

        for i, task_data_name in enumerate(direct_target_cols):
            ax = axes[i]
            y_true_task = y_test_direct[:, i]
            y_pred_task = predictions_direct[:, i]

            ax.scatter(y_true_task, y_pred_task, alpha=0.5, label="Predictions")

            # Determine plot limits to ensure 1:1 line covers data range
            min_val = min(np.min(y_true_task), np.min(y_pred_task))
            max_val = max(np.max(y_true_task), np.max(y_pred_task))
            # Add a small buffer to the limits
            buffer = (max_val - min_val) * 0.1
            ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], "r--", label="Perfect Prediction")
            ax.set_xlim(min_val - buffer, max_val + buffer)
            ax.set_ylim(min_val - buffer, max_val + buffer)

            ax.set_title(f"Task: {task_data_name}")
            ax.set_xlabel(f"Actual {task_data_name}")
            ax.set_ylabel(f"Predicted {task_data_name}")
            ax.legend()
            ax.grid(True)

            metrics = metrics_by_task[task_data_name]
            metrics_text = (
                f"MAE: {metrics['MAE']:.4f}\n"
                f"MSE: {metrics['MSE']:.4f}\n"
                f"RMSE: {metrics['RMSE']:.4f}\n"
                f"R²: {metrics['R2']:.4f}"
            )
            ax.text(
                0.05,
                0.95,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
            )
        
        # Add subplot for calculated Dissolved CO2
        ax_co2 = axes[num_direct_tasks] # Index for the new subplot
        ax_co2.scatter(y_test_indirect_co2, calculated_dissolved_co2, alpha=0.5, label="Calculated Predictions")
        
        min_val_co2 = min(np.min(y_test_indirect_co2), np.min(calculated_dissolved_co2))
        max_val_co2 = max(np.max(y_test_indirect_co2), np.max(calculated_dissolved_co2))
        buffer_co2 = (max_val_co2 - min_val_co2) * 0.1
        ax_co2.plot([min_val_co2 - buffer_co2, max_val_co2 + buffer_co2], [min_val_co2 - buffer_co2, max_val_co2 + buffer_co2], "r--", label="Perfect Prediction")
        ax_co2.set_xlim(min_val_co2 - buffer_co2, max_val_co2 + buffer_co2)
        ax_co2.set_ylim(min_val_co2 - buffer_co2, max_val_co2 + buffer_co2)

        ax_co2.set_title(f"Task: {indirect_target_col_for_eval} (Calculated)")
        ax_co2.set_xlabel(f"Actual {indirect_target_col_for_eval}")
        ax_co2.set_ylabel(f"Calculated {indirect_target_col_for_eval}")
        ax_co2.legend()
        ax_co2.grid(True)

        metrics_text_co2 = (
            f"MAE: {mae_co2:.4f}\n"
            f"MSE: {mse_co2:.4f}\n"
            f"RMSE: {rmse_co2:.4f}\n"
            f"R²: {r2_co2:.4f}"
        )
        ax_co2.text(
            0.05,
            0.95,
            metrics_text_co2,
            transform=ax_co2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7),
        )
        
        # Hide unused subplots if total_plots is not a perfect multiple of n_cols
        for i in range(total_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        eval_plot_path = os.path.join(output_dir, "multitask_evaluation_plots.png")
        plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Evaluation plots saved to {eval_plot_path}")

    return metrics_by_task

def evaluate_multitask_pressure_response(
    model_path: str,
    data_params: Dict[str, Any],
    multitask_params: Dict[str, Any],
    temperature: float,
    pressure_bounds: Tuple[float, float],
    fixed_ion_moles: Dict[str, float], # Fixed concentrations for ions
    fixed_co2_mol_kg: float, # This value is used for filtering experimental data only
    num_points: int = 100,
):
    """
    Evaluates the MultiTask model's predictions for each property
    across a range of pressures, while keeping other input features fixed.
    Generates separate plots for each predicted property (direct and calculated CO2).
    """
    output_dir = multitask_params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logging.info(
        f"Evaluating MultiTask properties vs pressure at {temperature}K. "
        f"Experimental Dissolved CO2 data filtered for approx {fixed_co2_mol_kg} mol/kg from: {model_path}"
    )

    feature_cols = data_params["feature_columns"]
    direct_target_cols_data = data_params["direct_target_columns"] # Direct target names as in the data file
    indirect_target_col_for_eval = data_params["indirect_target_column_for_eval"]
    model_task_names = [TASK_NAME_MAP[col] for col in direct_target_cols_data] # Task names as in the model

    min_p, max_p = pressure_bounds
    pressures = np.linspace(min_p, max_p, num_points)

    feature_indices = {col: i for i, col in enumerate(feature_cols)}

    scaler_path = data_params["scaler_path"]
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path} for pressure evaluation.")
    else:
        logging.warning(
            f"Scaler not found at {scaler_path}. Cannot scale data for pressure evaluation. "
            "Predictions might be inaccurate if the model was trained with scaled data."
        )
        scaler = None # Set to None if not found

    input_dim = len(feature_cols)
    model = MultiTaskNN(
        input_dim=input_dim,
        trunk_hidden_dims=multitask_params["trunk_hidden_dims"],
        head_hidden_dims=multitask_params["head_hidden_dims"],
        tasks=model_task_names
    )

    try:
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set model to evaluation mode
    except FileNotFoundError:
        logging.error(
            f"Model file not found at {model_path}. Cannot evaluate pressure sensitivity."
        )
        return
    except Exception as e:
        logging.error(f"Error loading model state_dict from {model_path}: {e}")
        return

    # Store predictions for all direct tasks and also for calculated CO2
    all_direct_predictions = np.zeros((num_points, len(model_task_names)))
    calculated_co2_predictions_for_plot = np.zeros(num_points) # For calculated CO2

    with torch.no_grad():
        for i, pressure_val in enumerate(pressures):
            features_instance = np.zeros((1, input_dim)) # Create a single instance for prediction

            # Set the values for all input features (Temperature, Pressure, Ions)
            if "Temperature (K)" in feature_indices:
                features_instance[0, feature_indices["Temperature (K)"]] = temperature
            if "Pressure (MPa)" in feature_indices:
                features_instance[0, feature_indices["Pressure (MPa)"]] = pressure_val
            
            # Set fixed ion concentrations
            for ion, value in fixed_ion_moles.items():
                if ion in feature_indices:
                    features_instance[0, feature_indices[ion]] = value
                else:
                    logging.warning(f"Fixed ion '{ion}' not found in feature columns. Skipping.")
            
            # Dissolved CO2 (mol/kg) is NOT an input feature, so it's not set here.

            # Scale features if scaler is available
            if scaler:
                features_instance_scaled = scaler.transform(features_instance)
            else:
                features_instance_scaled = features_instance

            X_tensor = torch.tensor(features_instance_scaled, dtype=torch.float32)
            direct_preds = model(X_tensor).cpu().numpy()[0, :] # Shape (n_tasks,) for this single instance
            all_direct_predictions[i, :] = direct_preds

            # Calculate Dissolved CO2 for this point using the predicted direct outputs and current pressure
            predicted_fugacity = direct_preds[model_task_names.index("fugacity")]
            predicted_mu = direct_preds[model_task_names.index("mu")]
            predicted_activity = direct_preds[model_task_names.index("activity")]
            predicted_xgas = direct_preds[model_task_names.index("xgas")]

            calculated_co2_predictions_for_plot[i] = calculate_dissolved_co2(
                fugacity=predicted_fugacity,
                mu=predicted_mu,
                activity=predicted_activity,
                xgas=predicted_xgas,
                pressure_MPa=pressure_val # Use the current simulated pressure
            )

    # Load experimental data once for all plots to avoid redundant reads
    try:
        exp_df = pd.read_csv(data_params["experimental_data_path"])
        exp_data_loaded = True
    except Exception as e:
        logging.warning(f"Could not load experimental data from {data_params['experimental_data_path']}: {e}")
        exp_data_loaded = False

    # Plotting loop: First for direct outputs
    for j, task_model_name in enumerate(model_task_names):
        task_data_name = INVERSE_TASK_NAME_MAP[task_model_name] # Get the original name from data for plot titles

        plt.figure(figsize=(10, 6))
        plt.plot(
            pressures,
            all_direct_predictions[:, j], # Select predictions for the current task
            'b-',
            linewidth=2,
            label=f"Predicted {task_data_name}",
        )
        plt.xlabel("Pressure (MPa)")
        plt.ylabel(f'Predicted {task_data_name}')
        plt.title(
            f"MultiTask NN: {task_data_name} vs Pressure at {temperature} K\n"
            f"(Fixed CO2 for Exp. Data Overlay: ~{fixed_co2_mol_kg:.2f} mol/kg)"
        )
        plt.grid(True)

        # Overlay experimental data for this specific task
        if exp_data_loaded:
            try:
                # Filter experimental data for matching conditions (temp, ions, fixed CO2 for plotting overlay)
                mask = np.isclose(exp_df["Temperature (K)"], temperature, atol=1)
                # Filter by fixed_co2_mol_kg for experimental data matching
                if indirect_target_col_for_eval in exp_df.columns:
                    mask &= np.isclose(exp_df[indirect_target_col_for_eval], fixed_co2_mol_kg, atol=0.01)
                for ion, val in fixed_ion_moles.items():
                    if ion in feature_indices:
                        mask &= np.isclose(exp_df[ion], val, atol=0.05)

                exp_df_filtered = exp_df[mask]
                if not exp_df_filtered.empty and task_data_name in exp_df_filtered.columns:
                    plt.scatter(
                        exp_df_filtered["Pressure (MPa)"],
                        exp_df_filtered[task_data_name], # Use the correct target column from experimental data
                        c="k",
                        marker="x",
                        label="Experimental",
                    )
            except Exception as e:
                logging.warning(f"Could not overlay experimental data for {task_data_name}: {e}")

        plt.legend()

        # Generate a unique filename for each plot based on task, temperature, and fixed CO2 (for exp. data)
        filename_base = f"multitask_pressure_response_direct_{task_model_name}_{temperature}K_targetCO2_{fixed_co2_mol_kg:.2f}"
        ion_str_parts = [
            f"{ion.replace('+','p').replace('-','m').replace('2','')}{value:.1f}"
            for ion, value in fixed_ion_moles.items()
            if value > 0 # Only include non-zero ion values in filename
        ]
        if ion_str_parts:
            filename_base += "_" + "_".join(ion_str_parts)

        plot_save_path = os.path.join(output_dir, f"{filename_base}.png")
        plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Pressure evaluation plot for {task_data_name} saved to {plot_save_path}")

        csv_save_path = os.path.join(output_dir, f"{filename_base}.csv")
        results_df = pd.DataFrame(
            {
                "Pressure (MPa)": pressures,
                f'Predicted {task_data_name}': all_direct_predictions[:, j],
            }
        )
        results_df.to_csv(csv_save_path, index=False)
        logging.info(f"Pressure evaluation CSV for {task_data_name} saved to {csv_save_path}")


    # Now, plot for calculated Dissolved CO2
    plt.figure(figsize=(10, 6))
    plt.plot(
        pressures,
        calculated_co2_predictions_for_plot,
        'b-',
        linewidth=2,
        label=f"Calculated {indirect_target_col_for_eval}",
    )
    plt.xlabel("Pressure (MPa)")
    plt.ylabel(f'Calculated {indirect_target_col_for_eval}')
    plt.title(
        f"MultiTask NN: Calculated {indirect_target_col_for_eval} vs Pressure at {temperature} K\n"
        f"(Experimental Data Filtered for: ~{fixed_co2_mol_kg:.2f} mol/kg)"
    )
    plt.grid(True)

    # Overlay experimental data for Dissolved CO2
    if exp_data_loaded:
        try:
            mask = np.isclose(exp_df["Temperature (K)"], temperature, atol=1)
            # Now, fixed_co2_mol_kg acts as the filter for the *actual* CO2 data points.
            if indirect_target_col_for_eval in exp_df.columns:
                mask &= np.isclose(exp_df[indirect_target_col_for_eval], fixed_co2_mol_kg, atol=0.01)
            for ion, val in fixed_ion_moles.items():
                if ion in feature_indices:
                    mask &= np.isclose(exp_df[ion], val, atol=0.05)

            exp_df_filtered = exp_df[mask]
            if not exp_df_filtered.empty and indirect_target_col_for_eval in exp_df_filtered.columns:
                plt.scatter(
                    exp_df_filtered["Pressure (MPa)"],
                    exp_df_filtered[indirect_target_col_for_eval],
                    c="k",
                    marker="x",
                    label="Experimental",
                )
        except Exception as e:
            logging.warning(f"Could not overlay experimental data for {indirect_target_col_for_eval}: {e}")

    plt.legend()
    
    filename_base_co2 = f"multitask_pressure_response_calculated_CO2_{temperature}K_expCO2_{fixed_co2_mol_kg:.2f}"
    ion_str_parts = [
        f"{ion.replace('+','p').replace('-','m').replace('2','')}{value:.1f}"
        for ion, value in fixed_ion_moles.items()
        if value > 0
    ]
    if ion_str_parts:
        filename_base_co2 += "_" + "_".join(ion_str_parts)

    plot_save_path_co2 = os.path.join(output_dir, f"{filename_base_co2}.png")
    plt.savefig(plot_save_path_co2, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Pressure evaluation plot for calculated Dissolved CO2 saved to {plot_save_path_co2}")

    csv_save_path_co2 = os.path.join(output_dir, f"{filename_base_co2}.csv")
    results_df_co2 = pd.DataFrame(
        {
            "Pressure (MPa)": pressures,
            f'Calculated {indirect_target_col_for_eval}': calculated_co2_predictions_for_plot,
        }
    )
    results_df_co2.to_csv(csv_save_path_co2, index=False)
    logging.info(f"Pressure evaluation CSV for calculated Dissolved CO2 saved to {csv_save_path_co2}")


if __name__ == "__main__":
    # --- Dummy Data Generation for Demonstration ---
    # This block creates placeholder CSV files if they don't exist.
    # In a real application, you would ensure your actual data files are at these paths.
    for path_key in ["experimental_train_path", "experimental_test_path", "experimental_data_path"]:
        file_path = data_params_dict[path_key]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            logging.info(f"Creating dummy data file: {file_path}")
            num_samples = 100 if "train" in path_key else 20
            dummy_data = {
                "Temperature (K)": np.random.uniform(273, 373, num_samples),
                "Pressure (MPa)": np.random.uniform(0.1, 10, num_samples),
                "Na+": np.random.uniform(0, 1, num_samples),
                "K+": np.random.uniform(0, 0.1, num_samples),
                "Mg+2": np.random.uniform(0, 0.05, num_samples),
                "Ca+2": np.random.uniform(0, 0.05, num_samples),
                "SO4-2": np.random.uniform(0, 0.5, num_samples),
                "Cl-": np.random.uniform(0, 1.5, num_samples),
                # Direct outputs (targets for training)
                "Fugacity": np.random.uniform(0.1, 10, num_samples),
                "ChemicalPotential": np.random.uniform(-10, -5, num_samples),
                "Activity": np.random.uniform(0.001, 1, num_samples),
                "MolarFraction": np.random.uniform(0.0001, 0.01, num_samples),
                # Indirect output (target for evaluation)
                "Dissolved CO2 (mol/kg)": np.random.uniform(0.01, 0.5, num_samples),
            }
            pd.DataFrame(dummy_data).to_csv(file_path, index=False)
    # --- End Dummy Data Generation ---

    best_model_file_path = train_multitask_model()

    if best_model_file_path and os.path.exists(best_model_file_path):
        print("\n--- Evaluating model on test data ---")
        evaluate_trained_multitask_model(model_path=best_model_file_path, plot=True)

        print("\n--- Evaluating model response to pressure ---")
        example_temperature = 304.15 # Example temperature in K
        example_pressure_bounds = (0.1, 10.0) # Example pressure range in MPa
        example_fixed_co2_mol_kg = 0.1 # Example fixed dissolved CO2 concentration (used for filtering experimental data only)
        
        # Collect all ion feature columns from data_params_dict's feature_columns
        # Exclude Temperature and Pressure as they are varied or explicitly set
        all_ion_keys = [
            col
            for col in data_params_dict["feature_columns"]
            if col not in ["Temperature (K)", "Pressure (MPa)"]
        ]
        # Example fixed ion concentrations (all set to 0.0 for simplicity, adjust as needed)
        example_fixed_ion_moles = {ion: 0.0 for ion in all_ion_keys}

        evaluate_multitask_pressure_response(
            model_path=best_model_file_path,
            data_params=data_params_dict,
            multitask_params=multitask_params_dict,
            temperature=example_temperature,
            pressure_bounds=example_pressure_bounds,
            fixed_ion_moles=example_fixed_ion_moles,
            fixed_co2_mol_kg=example_fixed_co2_mol_kg, # Passed as value to filter experimental data
            num_points=100
        )
    else:
        print(
            "Training did not complete successfully or model file not found. Skipping evaluations."
        )