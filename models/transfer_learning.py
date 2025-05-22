import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import joblib
import os
import logging
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import copy

# --- Data Parameters ---
data_params_dict = {
    "simulation_data_path": "data/processed/simulation.csv",
    "experimental_data_path": "data/processed/experimental.csv",
    "experimental_train_path": "data/processed/experimental_train.csv",
    "experimental_test_path": "data/processed/experimental_test.csv",
    "feature_columns": [
        "Temperature (K)",
        "Pressure (MPa)",
        "Na+", "K+", "Mg+2", "Ca+2", "SO4-2", "Cl-",
    ],
    "target_column": "Dissolved CO2 (mol/kg)",
    "scaler_path": "data/processed/feature_scaler.joblib",
}

# --- Transfer Learning Parameters ---
# Adjusted for a single fine-tuning stage
transfer_learning_params_dict = {
    "hidden_dims": [128, 64, 32],  # [FE_h1, FE_h2/TS_h1, TS_h2]
    "output_dir": "results/transfer_learning_model", # Changed output dir name
    "device": "cpu",

    # Pre-training parameters
    "pretrain_epochs": 200,
    "pretrain_batch_size": 128,
    "pretrain_learning_rate": 0.001,
    "pretrain_weight_decay": 0.0,
    "pretrain_use_early_stopping": True,
    "pretrain_early_stopping_patience": 10,
    "pretrain_early_stopping_delta": 0.0,
    "source_data_train_path_key": "simulation_data_path",
    "source_data_val_path_key": None, # Use source_data_train for val if None

    # Single Fine-tuning Stage parameters
    "finetune_epochs": 100,
    "finetune_batch_size": 32,
    "finetune_learning_rate": 0.0001, # Learning rate for fine-tuning
    "finetune_weight_decay": 0.0,
    "finetune_use_early_stopping": True, # DEMO: disabled
    "finetune_early_stopping_patience": 10,
    "finetune_early_stopping_delta": 0.0,
    "finetune_num_linear_layers_to_tune": 2, # Number of final linear layers to unfreeze and tune (e.g., 2 for task_specific head)

    "target_data_train_path_key": "experimental_train_path",
    "target_data_val_path_key": "experimental_test_path",

    "smoothness_weight": 0.0, # DEMO: disabled
    "smoothness_temperature": 304.0,
    "smoothness_ion_moles": {
        ion: 0.0
        for ion in data_params_dict["feature_columns"]
        if ion not in ["Temperature (K)", "Pressure (MPa)"]
    },
    "smoothness_pressure_bounds": (0.1, 100.0),
    "smoothness_num_points": 50,
}

# --- Model Definition ---
class TransferLearningNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
        super(TransferLearningNN, self).__init__()
        if len(hidden_dims) < 3:
            raise ValueError("hidden_dims must have at least 3 elements for this TransferLearningNN structure: [FE_h1, FE_h2/TS_h1, TS_h2]")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Feature Extractor: 2 linear layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )
        # Task-Specific Head: 2 linear layers
        self.task_specific = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        output = self.task_specific(features)
        return output

    def _get_linear_layers(self) -> List[nn.Linear]:
        """Helper to get all nn.Linear layers in order from input to output."""
        linear_layers = []
        for module in [self.feature_extractor, self.task_specific]:
            for layer_component in module: # nn.Sequential is iterable
                if isinstance(layer_component, nn.Linear):
                    linear_layers.append(layer_component)
        return linear_layers

    def unfreeze_all_layers(self):
        """Sets requires_grad = True for all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = True
        logging.info("All model layers unfrozen.")

    def set_trainable_final_linear_layers(self, num_layers_to_train: int):
        """
        Freezes all model parameters, then unfreezes the last 'num_layers_to_train'
        nn.Linear layers.
        """
        # First, freeze all parameters in the model
        for param in self.parameters():
            param.requires_grad = False
        
        linear_layers = self._get_linear_layers()
        total_linear_layers = len(linear_layers)
        
        if num_layers_to_train > total_linear_layers:
            logging.warning(
                f"Requested to train {num_layers_to_train} final linear layers, "
                f"but model only has {total_linear_layers}. Training all {total_linear_layers} linear layers."
            )
            num_layers_to_train = total_linear_layers
        
        if num_layers_to_train < 0:
            logging.warning(f"num_layers_to_train ({num_layers_to_train}) cannot be negative. Setting to 0 (all linear layers frozen).")
            num_layers_to_train = 0

        # Unfreeze the parameters of the last 'num_layers_to_train' linear layers
        if num_layers_to_train > 0:
            for i in range(num_layers_to_train):
                layer_to_unfreeze = linear_layers[-(i + 1)] # Access from the end
                for param in layer_to_unfreeze.parameters():
                    param.requires_grad = True
        
        if num_layers_to_train == 0:
             logging.info(f"All {total_linear_layers} linear layers are frozen.")
        else:
            logging.info(
                f"Set last {num_layers_to_train} out of {total_linear_layers} "
                f"linear layers to be trainable for fine-tuning."
            )
            
    def get_trainable_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# --- Dataset Class ---
class TabularDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        if len(self.targets.shape) == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

# --- Utility Functions ---
def setup_logging(log_path: str):
    # Remove existing handlers to avoid duplicate logs if re-running in same session
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

def load_data(
    data_path: str,
    feature_cols: List[str],
    target_col: str,
    scaler: Any = None,
    is_target_data: bool = False # True if loading target domain data (experimental)
) -> Tuple[np.ndarray, np.ndarray, Any]: # Adjusted return type, scaler is always returned
    if not os.path.exists(data_path):
        logging.warning(f"Data file not found: {data_path}. Generating dummy data.")
        num_dummy_samples = 100 if "train" in data_path or "simulation" in data_path else 30
        X = np.random.rand(num_dummy_samples, len(feature_cols))
        y = np.random.rand(num_dummy_samples)
        
        if scaler is None and (is_target_data or "simulation" in data_path): # Fit scaler if not provided
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Attempt to save dummy scaler, might be overwritten if real data is processed later
            if not os.path.exists(data_params_dict["scaler_path"]) or is_target_data: # Prioritize fitting on source
                try:
                    joblib.dump(scaler, data_params_dict["scaler_path"])
                    logging.info(f"Fitted and saved a dummy scaler to {data_params_dict['scaler_path']}")
                except Exception as e:
                    logging.error(f"Could not save dummy scaler: {e}")
        elif scaler is not None:
             X_scaled = scaler.transform(X)
        else: # Scaler is None and not fitting (should not happen with above logic for train/source)
            X_scaled = X
        return X_scaled, y, scaler

    df = pd.read_csv(data_path)
    X = df[feature_cols].values
    y = df[target_col].values

    if scaler:
        X_scaled = scaler.transform(X)
    else: # Scaler not provided
        if is_target_data: # If target data and no scaler, means pre-training didn't fit one (e.g. source data was pre-scaled)
            logging.warning("Scaler not provided for target data. This is unusual. Attempting to load or fit.")
            if os.path.exists(data_params_dict["scaler_path"]): # Try to load if path exists
                 scaler = joblib.load(data_params_dict["scaler_path"])
                 X_scaled = scaler.transform(X)
            else: # Fit on this target data as last resort
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                joblib.dump(scaler, data_params_dict["scaler_path"])
                logging.info(f"Fitted scaler on target data '{data_path}' and saved to {data_params_dict['scaler_path']}.")
        else: # Source data (pre-training), fit scaler if not provided
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            joblib.dump(scaler, data_params_dict["scaler_path"])
            logging.info(f"Fitted scaler on source data '{data_path}' and saved to {data_params_dict['scaler_path']}")
    
    return X_scaled, y, scaler


# Placeholder for smoothness metric calculation
def calculate_smoothness_metric(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """
    Calculates a smoothness metric. A simple version is the mean absolute change in slope.
    Args:
        x_values: Input values (e.g., pressure), assumed to be sorted or sortable.
        y_values: Output values (e.g., model predictions).
    Returns:
        A scalar representing the non-smoothness (penalty).
    """
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return 0.0
    
    # Ensure x_values are sorted for correct slope calculation
    # This is important if data isn't inherently ordered (e.g. batch from DataLoader)
    # However, for the smoothness grid, x_values (pressures_smooth) are already sorted.
    # For safety, we can re-sort if needed, but for fixed grid it's not.
    # sorted_indices = np.argsort(x_values)
    # x_sorted = x_values[sorted_indices]
    # y_sorted = y_values[sorted_indices]
    x_sorted, y_sorted = x_values, y_values # Assuming x_values are already sorted for this specific use case

    if len(x_sorted) < 3: # Need at least 3 points to calculate change in slope
        return 0.0

    # Calculate slopes; add a small epsilon to avoid division by zero if x points are identical
    dx = np.diff(x_sorted)
    dy = np.diff(y_sorted)
    
    # Filter out points where dx is zero or too small
    valid_dx_indices = np.where(dx > 1e-9)[0]
    if len(valid_dx_indices) < 1: # Not enough valid segments for slopes
        return 0.0
        
    slopes = dy[valid_dx_indices] / dx[valid_dx_indices]
    
    if len(slopes) < 2: # Need at least 2 slopes to calculate difference in slopes
        return 0.0
        
    # Mean absolute difference of consecutive slopes
    # This penalizes rapid changes in the gradient
    smoothness_penalty = np.mean(np.abs(np.diff(slopes)))
    return float(smoothness_penalty)


def _train_stage(
    model: TransferLearningNN,
    stage_name: str,
    train_data_path_key: str,
    val_data_path_key: str, # Can be the same as train for pre-training if no separate val
    stage_params: Dict[str, Any],
    global_data_params: Dict[str, Any],
    global_tl_params: Dict[str, Any],
    scaler: Any # Pass scaler, it might be updated
) -> Tuple[str, TransferLearningNN, Any]: # Return scaler

    output_dir = global_tl_params["output_dir"]
    device = torch.device(global_tl_params["device"])
    model.to(device)

    logging.info(f"--- Starting Training Stage: {stage_name} ---")
    logging.info(f"Trainable parameters: {model.get_trainable_parameters()}")

    feature_cols = global_data_params["feature_columns"]
    target_col = global_data_params["target_column"]

    train_path = global_data_params[train_data_path_key]
    logging.info(f"Loading training data from: {train_path}")
    is_target_stage = "finetune" in stage_name.lower() # For load_data logic
    X_train_scaled, y_train, scaler = load_data(train_path, feature_cols, target_col, scaler, is_target_data=is_target_stage)
    if scaler is None: # Should not happen if load_data works correctly
        logging.error("Scaler is None after loading training data. This is problematic.")
        raise ValueError("Scaler cannot be None after data loading.")
    logging.info("Scaler active for this stage.")


    if val_data_path_key and global_data_params.get(val_data_path_key):
        val_path = global_data_params[val_data_path_key]
        logging.info(f"Loading validation data from: {val_path}")
        X_val_scaled, y_val, _ = load_data(val_path, feature_cols, target_col, scaler, is_target_data=True) # Val data is always target-like context
    else: 
        logging.info(f"No separate validation data path for '{stage_name}'. Using training data for validation metrics.")
        X_val_scaled, y_val = X_train_scaled, y_train


    train_dataset = TabularDataset(X_train_scaled, y_train)
    val_dataset = TabularDataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=stage_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=stage_params["batch_size"], shuffle=False)
    logging.info(f"Stage {stage_name}: Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage_params["learning_rate"],
        weight_decay=stage_params.get("weight_decay", 0.0)
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = stage_params.get("early_stopping_patience", 10)
    delta = stage_params.get("early_stopping_delta", 0.0)
    use_early_stopping = stage_params.get("use_early_stopping", False)
    
    model_filename = f"best_model_{stage_name.lower().replace(' ', '_').replace('-', '_')}.pt"
    model_save_path = os.path.join(output_dir, model_filename)

    train_losses, val_losses = [], []

    for epoch in range(stage_params["epochs"]):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            
            smooth_w = global_tl_params.get("smoothness_weight", 0.0)
            if smooth_w > 0 and ("finetune" in stage_name.lower() or "pretrain_smooth" in stage_name.lower()):
                pb_min, pb_max = global_tl_params.get("smoothness_pressure_bounds", (0.1, 100.0))
                n_pts = global_tl_params.get("smoothness_num_points", 50)
                pressures_smooth = np.linspace(pb_min, pb_max, n_pts)
                
                grid = np.zeros((n_pts, len(feature_cols)))
                feature_indices = {col: i for i, col in enumerate(feature_cols)}

                if "Temperature (K)" in feature_indices:
                    grid[:, feature_indices["Temperature (K)"]] = global_tl_params.get("smoothness_temperature", 298.15)
                for ion, val_ion in global_tl_params.get("smoothness_ion_moles", {}).items():
                    if ion in feature_indices:
                        grid[:, feature_indices[ion]] = val_ion # Corrected variable name
                if "Pressure (MPa)" in feature_indices:
                    grid[:, feature_indices["Pressure (MPa)"]] = pressures_smooth
                
                grid_scaled = scaler.transform(grid) # Scaler must exist here
                
                current_mode_is_train = model.training
                model.eval() # For consistent predictions during smoothness calculation
                with torch.no_grad():
                    smooth_preds = model(torch.tensor(grid_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()
                if current_mode_is_train:
                    model.train()

                smoothness_penalty_val = calculate_smoothness_metric(pressures_smooth, smooth_preds)
                loss += smooth_w * smoothness_penalty_val
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs_val, targets_val in val_loader:
                inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                outputs_val = model(inputs_val)
                val_loss_item = criterion(outputs_val, targets_val)
                val_running_loss += val_loss_item.item() * inputs_val.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        logging.info(
            f"Stage {stage_name} - Epoch {epoch+1}/{stage_params['epochs']}, Train Loss: {epoch_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss - delta:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
            logging.info(f"Validation loss improved. Saving model to {model_save_path}")
        else:
            epochs_no_improve += 1

        if use_early_stopping and epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered for stage {stage_name} after {epoch+1} epochs.")
            break
    
    plt.figure(figsize=(10, 6))
    epochs_range = list(range(1, len(train_losses) + 1))
    plt.plot(epochs_range, train_losses, label="Training Loss", marker="o", linestyle="-", color="blue")
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker="x", linestyle="-", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title(f"{stage_name} - Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    if val_losses:
        min_val_loss_value = min(val_losses) if val_losses else float('nan')
        if not np.isnan(min_val_loss_value):
            best_epoch_idx = val_losses.index(min_val_loss_value)
            best_epoch_num = epochs_range[best_epoch_idx]
            plt.annotate(
                f"Best val loss: {min_val_loss_value:.6f} at epoch {best_epoch_num}",
                xy=(best_epoch_num, min_val_loss_value),
                xytext=(best_epoch_num + max(1, len(epochs_range)*0.05), min_val_loss_value * 1.05 if min_val_loss_value != 0 else 0.1),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5)
            )
    plot_path = os.path.join(output_dir, f"loss_plot_{stage_name.lower().replace(' ', '_').replace('-', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Loss plot for stage {stage_name} saved to {plot_path}")

    logging.info(f"Stage {stage_name} finished. Best validation loss: {best_val_loss:.6f}")
    
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        logging.info(f"Loaded best model weights from {model_save_path} for stage {stage_name}.")
    else:
        logging.warning(f"Could not find best model at {model_save_path}. Using current model state for stage {stage_name}.")

    return model_save_path, model, scaler


def train_and_evaluate_transfer_model():
    tl_params = transfer_learning_params_dict
    data_params = data_params_dict
    output_dir = tl_params["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "transfer_training.log") # New log file name
    setup_logging(log_file)
    logging.info("Starting Transfer Learning NN training (single fine-tuning stage) and evaluation...")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Using device: {tl_params['device']}")

    input_dim = len(data_params["feature_columns"])
    
    scaler = None
    if os.path.exists(data_params["scaler_path"]):
        try:
            scaler = joblib.load(data_params["scaler_path"])
            logging.info(f"Loaded existing scaler from {data_params['scaler_path']}")
        except Exception as e:
            logging.error(f"Error loading scaler from {data_params['scaler_path']}: {e}. Will attempt to fit.")
    else:
        logging.info(f"Scaler not found at {data_params['scaler_path']}. Will be fitted on source data or first available data.")

    current_model = TransferLearningNN(
        input_dim=input_dim,
        hidden_dims=tl_params["hidden_dims"],
        output_dim=1
    )
    
    # --- Stage 1: Pre-training ---
    current_model.unfreeze_all_layers() # Ensure all params are trainable for pre-training
    pretrain_stage_params = {
        "epochs": tl_params["pretrain_epochs"],
        "batch_size": tl_params["pretrain_batch_size"],
        "learning_rate": tl_params["pretrain_learning_rate"],
        "weight_decay": tl_params["pretrain_weight_decay"],
        "use_early_stopping": tl_params["pretrain_use_early_stopping"],
        "early_stopping_patience": tl_params["pretrain_early_stopping_patience"],
        "early_stopping_delta": tl_params["pretrain_early_stopping_delta"],
    }
    _, current_model, scaler = _train_stage(
        model=current_model,
        stage_name="Pre-training",
        train_data_path_key=tl_params["source_data_train_path_key"],
        val_data_path_key=tl_params.get("source_data_val_path_key", tl_params["source_data_train_path_key"]),
        stage_params=pretrain_stage_params,
        global_data_params=data_params,
        global_tl_params=tl_params,
        scaler=scaler # Pass scaler, it will be fitted if None
    )
    
    # --- Stage 2: Single Fine-tuning Stage ---
    # Set trainable layers for fine-tuning based on configuration
    num_layers_to_tune = tl_params.get("finetune_num_linear_layers_to_tune", 1) # Default to 2 if not specified
    current_model.set_trainable_final_linear_layers(num_layers_to_tune)
    
    finetune_stage_params = {
        "epochs": tl_params["finetune_epochs"],
        "batch_size": tl_params["finetune_batch_size"],
        "learning_rate": tl_params["finetune_learning_rate"],
        "weight_decay": tl_params["finetune_weight_decay"],
        "use_early_stopping": tl_params["finetune_use_early_stopping"],
        "early_stopping_patience": tl_params["finetune_early_stopping_patience"],
        "early_stopping_delta": tl_params["finetune_early_stopping_delta"],
    }
    final_model_path, final_model, scaler = _train_stage(
        model=current_model,
        stage_name="Fine-tuning", # Single fine-tuning stage
        train_data_path_key=tl_params["target_data_train_path_key"],
        val_data_path_key=tl_params["target_data_val_path_key"],
        stage_params=finetune_stage_params,
        global_data_params=data_params,
        global_tl_params=tl_params,
        scaler=scaler # Pass the (potentially updated) scaler
    )
    
    logging.info("All training stages completed.")
    
    if final_model_path and os.path.exists(final_model_path):
        logging.info(f"\n--- Evaluating final fine-tuned model from: {final_model_path} ---")
        if scaler is None:
            logging.error("Scaler is None before evaluation. This indicates a problem in the training pipeline.")
        evaluate_trained_transfer_model(model_path=final_model_path, scaler=scaler, plot=True)

        logging.info("\n--- Evaluating final model response to pressure ---")
        evaluate_transfer_increasing_pressure(
            model_path=final_model_path,
            scaler=scaler,
            temperature=tl_params["smoothness_temperature"],
            pressure_bounds=tl_params["smoothness_pressure_bounds"],
            ion_moles=tl_params["smoothness_ion_moles"],
            num_points=tl_params["smoothness_num_points"]
        )
    else:
        logging.error("Final model path not found or training failed. Skipping final evaluations.")

    return final_model_path


def evaluate_trained_transfer_model(model_path: str, scaler: Any, plot: bool = False):
    model_params = transfer_learning_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]
    device = torch.device(model_params["device"])

    if scaler is None:
        logging.error("Scaler is None during evaluation. Cannot proceed.")
        return None, None, None # Or raise error

    logging.info(f"Evaluating Transfer Learning model from: {model_path}")

    val_data_key = model_params["target_data_val_path_key"]
    X_test_scaled, y_test, _ = load_data( # Use load_data to handle dummy data if needed
        data_params[val_data_key],
        data_params["feature_columns"],
        data_params["target_column"],
        scaler,
        is_target_data=True
    )

    input_dim = len(data_params["feature_columns"])
    model = TransferLearningNN(
        input_dim=input_dim, hidden_dims=model_params["hidden_dims"], output_dim=1
    )
    model.to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model state_dict from {model_path}: {e}. Cannot evaluate.")
        return None, None, None

    predictions = []
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        preds_tensor = model(X_tensor)
        predictions = preds_tensor.cpu().numpy().flatten()

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    logging.info("Transfer Learning Model Evaluation Results (on target validation set):")
    logging.info(f"\tMAE: {mae:.4f}")
    logging.info(f"\tMSE: {mse:.4f}")
    logging.info(f"\tRMSE: {rmse:.4f}")
    logging.info(f"\tR²: {r2:.4f}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5, label="Predictions")
        max_val = max(np.max(y_test), np.max(predictions)) if len(y_test) > 0 and len(predictions) > 0 else 1
        min_val = min(np.min(y_test), np.min(predictions)) if len(y_test) > 0 and len(predictions) > 0 else 0
        plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
        plt.title(f"Transfer Learning NN: Actual vs Predicted {data_params['target_column']}")
        plt.xlabel(f"Actual {data_params['target_column']}")
        plt.ylabel(f"Predicted {data_params['target_column']}")
        plt.legend()
        plt.grid(True)
        metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, verticalalignment="top",
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
        eval_plot_path = os.path.join(output_dir, "transfer_model_evaluation_plot.png")
        plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Evaluation plot saved to {eval_plot_path}")
    return mae, mse, r2


def evaluate_transfer_increasing_pressure(
    model_path: str,
    scaler: Any,
    temperature: float,
    pressure_bounds: Tuple[float, float],
    ion_moles: Dict[str, float],
    num_points: int = 100,
    plot_color: str = "g-",
):
    model_params = transfer_learning_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]
    device = torch.device(model_params["device"])

    if scaler is None:
        logging.error("Scaler is None during pressure evaluation. Cannot proceed.")
        return None, None

    logging.info(
        f"Evaluating Transfer Learning CO2 solubility vs pressure at {temperature}K from: {model_path}"
    )

    feature_cols = data_params["feature_columns"]
    min_p, max_p = pressure_bounds
    pressures = np.linspace(min_p, max_p, num_points)

    feature_template = np.zeros((1, len(feature_cols)))
    feature_indices = {col: i for i, col in enumerate(feature_cols)}

    if "Temperature (K)" in feature_indices:
        feature_template[0, feature_indices["Temperature (K)"]] = temperature
    for ion, value in ion_moles.items():
        if ion in feature_indices:
            feature_template[0, feature_indices[ion]] = value
    
    input_dim = len(feature_cols)
    model = TransferLearningNN(
        input_dim=input_dim, hidden_dims=model_params["hidden_dims"], output_dim=1
    )
    model.to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model {model_path} for pressure sensitivity: {e}")
        return None, None

    pressure_predictions = []
    with torch.no_grad():
        for p_val in pressures:
            current_features = feature_template.copy()
            if "Pressure (MPa)" in feature_indices:
                current_features[0, feature_indices["Pressure (MPa)"]] = p_val
            
            current_features_scaled = scaler.transform(current_features)
            
            X_tensor = torch.tensor(current_features_scaled, dtype=torch.float32).to(device)
            pred = model(X_tensor).cpu().numpy().item()
            pressure_predictions.append(pred)
    
    pressure_predictions = np.array(pressure_predictions)
    # Calculate smoothness for the generated curve
    smooth_val = calculate_smoothness_metric(pressures, pressure_predictions) 

    plt.figure(figsize=(10, 6))
    plt.plot(pressures, pressure_predictions, plot_color, linewidth=2, 
             label=f"Transfer Model (Smoothness: {smooth_val:.5g})") # Use .5g for auto precision
    plt.xlabel("Pressure (MPa)")
    plt.ylabel(f'Predicted {data_params["target_column"]}')
    plt.title(f"Transfer Model: CO2 Solubility vs Pressure at {temperature} K")
    plt.grid(True)

    try: 
        exp_df_path = data_params.get("experimental_data_path", None)
        if exp_df_path and os.path.exists(exp_df_path):
            exp_df = pd.read_csv(exp_df_path)
            # Filter experimental data for conditions close to the plot's fixed parameters
            mask = np.isclose(exp_df["Temperature (K)"], temperature, atol=2.0) # Tolerance for temperature matching
            for ion, val_target in ion_moles.items():
                if ion in exp_df.columns:
                     # Relative tolerance for ion concentrations, handles zeros by adding small epsilon
                     ion_val_from_df = exp_df[ion]
                     atol_ion = 0.05 * (np.abs(val_target) + 1e-6) 
                     mask &= np.isclose(ion_val_from_df, val_target, atol=atol_ion)
            
            exp_subset = exp_df[mask]
            if not exp_subset.empty:
                plt.scatter(
                    exp_subset["Pressure (MPa)"],
                    exp_subset[data_params["target_column"]],
                    c="k", marker="x", s=50, label="Experimental Data (approx. conditions)"
                )
            else:
                logging.info(f"No experimental data found matching conditions for T={temperature}K and specified ion moles for overlay.")
        else:
            logging.info("Experimental data path not found or not specified, skipping overlay on pressure plot.")
    except Exception as e:
        logging.warning(f"Could not overlay experimental data for pressure plot: {e}")
    
    plt.legend()
    filename_base = f"transfer_co2_vs_pressure_{temperature}K"
    ion_str_parts = []
    # Create a string from non-zero ion concentrations for filename
    for ion,value in ion_moles.items():
        if ion in feature_indices and abs(value) > 1e-6: # Only include if ion is a feature and non-negligible
            # Sanitize ion name for filename
            sanitized_ion = ion.replace('+','p').replace('-','m').replace('(','').replace(')','').replace('/','_').replace('2','') 
            ion_str_parts.append(f"{sanitized_ion}{value:.2f}")

    if ion_str_parts:
        filename_base += "_" + "_".join(ion_str_parts)
    
    plot_save_path = os.path.join(output_dir, f"{filename_base}.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Pressure evaluation plot saved to {plot_save_path}")

    csv_save_path = os.path.join(output_dir, f"{filename_base}.csv")
    results_df = pd.DataFrame({
        "Pressure (MPa)": pressures,
        f'Predicted {data_params["target_column"]}': pressure_predictions
    })
    results_df.to_csv(csv_save_path, index=False)
    logging.info(f"Pressure evaluation CSV saved to {csv_save_path}")

    return pressures, pressure_predictions


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    feature_cols_main = data_params_dict["feature_columns"]
    target_col_main = data_params_dict["target_column"]
    
    dummy_files_to_create = {
        data_params_dict["simulation_data_path"]: 200,
        data_params_dict["experimental_train_path"]: 80,
        data_params_dict["experimental_test_path"]: 20,
        data_params_dict["experimental_data_path"]: 100
    }

    for f_path, n_samples in dummy_files_to_create.items():
        if not os.path.exists(f_path):
            print(f"Creating dummy data file: {f_path}")
            df_cols = feature_cols_main + [target_col_main]
            dummy_data = np.random.rand(n_samples, len(df_cols))
            
            temp_idx = feature_cols_main.index("Temperature (K)") if "Temperature (K)" in feature_cols_main else -1
            pressure_idx = feature_cols_main.index("Pressure (MPa)") if "Pressure (MPa)" in feature_cols_main else -1

            if temp_idx != -1:
                dummy_data[:, temp_idx] = np.random.uniform(280, 500, n_samples)
            if pressure_idx != -1:
                dummy_data[:, pressure_idx] = np.random.uniform(0.1, 150, n_samples)
            
            # Make target somewhat correlated for "better" dummy data
            target_val_idx = len(feature_cols_main) # Target is the last column
            if pressure_idx != -1 and temp_idx != -1:
                 dummy_data[:, target_val_idx] = \
                    0.01 * dummy_data[:, pressure_idx] - \
                    0.005 * dummy_data[:, temp_idx] + \
                    np.random.normal(0, 0.1, n_samples) + 0.5
                 dummy_data[:, target_val_idx] = np.maximum(0.01, dummy_data[:, target_val_idx]) # Ensure positive
            else: # Simpler target if T or P not present
                dummy_data[:, target_val_idx] = np.random.rand(n_samples) * 2.0


            pd.DataFrame(dummy_data, columns=df_cols).to_csv(f_path, index=False)

    if not os.path.exists(data_params_dict["scaler_path"]):
        print(f"Note: Scaler path {data_params_dict['scaler_path']} does not exist. It will be created during training.")

    final_model_tl_path = train_and_evaluate_transfer_model()
    
    if final_model_tl_path and os.path.exists(final_model_tl_path):
        print(f"\nTransfer learning (single fine-tune stage) process complete. Final model: {final_model_tl_path}")
    else:
        print("\nTransfer learning process completed, but final model path was not found or an issue occurred.")