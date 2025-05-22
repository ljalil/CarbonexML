import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import joblib
from Commons.smoothness_metric import calculate_smoothness_metric

import os
import logging
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data_params_dict = {
    "simulation_data_path": "data/processed/simulation.csv",
    "experimental_data_path": "data/processed/experimental.csv",
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
    "scaler_path": "data/processed/feature_scaler.joblib",
}

blackbox_params_dict = {
    "hidden_dims": [64, 128, 64],
    "output_dir": "results/blackbox_model",
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "use_early_stopping": True,
    "early_stopping_patience": 20,
    "early_stopping_delta": 0.0,
    "device": "cpu",
    "smoothness_weight": 0,
    "smoothness_temperature": 423.15,
    "smoothness_ion_moles": {
        ion: 0.0
        for ion in data_params_dict["feature_columns"]
        if ion not in ["Temperature (K)", "Pressure (MPa)"]
    },
    "smoothness_pressure_bounds": (0.0, 100.0),
    "smoothness_num_points": 100,
}


class BlackBoxNN(nn.Module):
    """
    A simple fully-connected neural network (MLP) for regression.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 1):
        super(BlackBoxNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TabularDataset(Dataset):
    """Simple Dataset for tabular data."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        if len(self.targets.shape) == 1:
            self.targets = self.targets.unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def setup_logging(log_path: str):
    """Sets up logging to file and console."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def train_blackbox_model() -> str:
    model_params = blackbox_params_dict
    data_params = data_params_dict

    output_dir = model_params["output_dir"]

    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    setup_logging(log_file)
    logging.info(f"Starting BlackBox NN training...")
    logging.info(f"Output directory: {output_dir}")

    logging.info("Loading data...")
    train_df = pd.read_csv(data_params["experimental_train_path"])
    test_df = pd.read_csv(data_params["experimental_test_path"])

    feature_cols = data_params["feature_columns"]
    target_col = data_params["target_column"]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = test_df[feature_cols].values
    y_val = test_df[target_col].values

    scaler = joblib.load(data_params["scaler_path"])
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
    model = BlackBoxNN(
        input_dim=input_dim,
        hidden_dims=model_params["hidden_dims"],
        output_dim=1,
    )

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=model_params["learning_rate"],
        weight_decay=model_params.get("weight_decay", 0),
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = model_params.get("early_stopping_patience", 10)
    model_save_path = os.path.join(output_dir, "best_blackbox_model.pt")

    train_losses = []
    val_losses = []

    logging.info("Starting training loop...")
    for epoch in range(model_params["epochs"]):
        model.train()
        running_loss = 0.0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        smooth_w = model_params.get("smoothness_weight", 0.0)
        if smooth_w > 0:
            pb_min, pb_max = model_params.get(
                "smoothness_pressure_bounds", (0.0, 100.0)
            )
            n_pts = model_params.get("smoothness_num_points", 100)
            pressures_smooth = np.linspace(pb_min, pb_max, n_pts)
            grid = np.zeros((n_pts, input_dim))
            if "Temperature (K)" in feature_cols:
                ti = feature_cols.index("Temperature (K)")
                grid[:, ti] = model_params.get("smoothness_temperature", 298.15)
            for ion, val in model_params.get("smoothness_ion_moles", {}).items():
                if ion in feature_cols:
                    grid[:, feature_cols.index(ion)] = val
            if "Pressure (MPa)" in feature_cols:
                pi = feature_cols.index("Pressure (MPa)")
                grid[:, pi] = pressures_smooth
            try:
                grid_scaled = scaler.transform(grid)
            except Exception:
                grid_scaled = grid
            with torch.no_grad():
                smooth_preds = (
                    model(torch.tensor(grid_scaled, dtype=torch.float32))
                    .cpu()
                    .numpy()
                    .flatten()
                )
            smooth_loss = calculate_smoothness_metric(pressures_smooth, smooth_preds)
            epoch_loss += smooth_w * smooth_loss
            logging.info(
                f"Epoch {epoch+1}, Train smoothness penalty: {smooth_w * smooth_loss:.6f}"
            )
        train_losses.append(epoch_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs, targets
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        smooth_w = model_params.get("smoothness_weight", 0.0)
        if smooth_w > 0:
            pb_min, pb_max = model_params.get(
                "smoothness_pressure_bounds", (0.0, 100.0)
            )
            n_pts = model_params.get("smoothness_num_points", 100)
            pressures_smooth = np.linspace(pb_min, pb_max, n_pts)
            grid = np.zeros((n_pts, input_dim))
            if "Temperature (K)" in feature_cols:
                ti = feature_cols.index("Temperature (K)")
                grid[:, ti] = model_params.get("smoothness_temperature", 298.15)
            for ion, val in model_params.get("smoothness_ion_moles", {}).items():
                if ion in feature_cols:
                    grid[:, feature_cols.index(ion)] = val
            if "Pressure (MPa)" in feature_cols:
                pi = feature_cols.index("Pressure (MPa)")
                grid[:, pi] = pressures_smooth
            try:
                grid_scaled = (
                    scaler.transform(grid)
                    if "scaler" in locals() and scaler is not None
                    else grid
                )
            except NameError:
                grid_scaled = grid
            with torch.no_grad():
                smooth_preds = (
                    model(torch.tensor(grid_scaled, dtype=torch.float32))
                    .cpu()
                    .numpy()
                    .flatten()
                )
            smooth_loss = calculate_smoothness_metric(pressures_smooth, smooth_preds)
            epoch_val_loss += smooth_w * smooth_loss
            logging.info(
                f"Epoch {epoch+1}, Smoothness penalty: {smooth_w * smooth_loss:.6f}"
            )

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
    plt.title("Training and Validation Loss")
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


def evaluate_trained_blackbox_model(model_path: str, plot: bool = False):
    """
    Evaluates a trained BlackBox model on the test dataset.
    Uses global blackbox_params_dict and data_params_dict.
    """
    model_params = blackbox_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]

    logging.info(f"Evaluating BlackBox model from: {model_path}")

    try:
        test_df = pd.read_csv(data_params["experimental_test_path"])
    except FileNotFoundError:
        logging.error(
            f"Test data file not found: {data_params['experimental_test_path']}"
        )
        return

    feature_cols = data_params["feature_columns"]
    target_col = data_params["target_column"]

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    scaler_path = data_params["scaler_path"]
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_test_scaled = scaler.transform(X_test)
        logging.info(f"Loaded scaler from {scaler_path} for evaluation.")
    else:
        logging.warning(
            f"Scaler not found at {scaler_path} for evaluation. Using unscaled data."
        )
        X_test_scaled = X_test

    input_dim = len(feature_cols)
    model = BlackBoxNN(
        input_dim=input_dim, hidden_dims=model_params["hidden_dims"], output_dim=1
    )

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Cannot evaluate.")
        return
    except Exception as e:
        logging.error(f"Error loading model state_dict from {model_path}: {e}")
        return

    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        predictions = model(X_tensor).cpu().numpy().flatten()

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    logging.info("BlackBox Model Evaluation Results:")
    logging.info(f"\tMAE: {mae:.4f}")
    logging.info(f"\tMSE: {mse:.4f}")
    logging.info(f"\tRMSE: {rmse:.4f}")
    logging.info(f"\tR²: {r2:.4f}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, alpha=0.5, label="Predictions")

        max_val = max(np.max(y_test), np.max(predictions))
        min_val = min(np.min(y_test), np.min(predictions))
        plt.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction"
        )

        plt.title("BlackBox NN: Actual vs Predicted CO2 Solubility")
        plt.xlabel(f"Actual {target_col}")
        plt.ylabel(f"Predicted {target_col}")
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

        eval_plot_path = os.path.join(output_dir, "blackbox_evaluation_plot.png")
        os.makedirs(os.path.dirname(eval_plot_path), exist_ok=True)
        plt.savefig(eval_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Evaluation plot saved to {eval_plot_path}")

    return mae, mse, r2


def evaluate_increasing_pressure(
    model_path: str,
    temperature: float,
    pressure_bounds: Tuple[float, float],
    ion_moles: Dict[str, float],
    num_points: int = 100,
    plot_color: str = "b-",
):
    """
    Evaluate the BlackBox model's predictions for CO2 solubility
    across a range of pressures at a fixed temperature and ion composition.
    Uses global blackbox_params_dict and data_params_dict.
    """
    model_params = blackbox_params_dict
    data_params = data_params_dict
    output_dir = model_params["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    logging.info(
        f"Evaluating BlackBox CO2 solubility vs pressure at {temperature}K from: {model_path}"
    )

    feature_cols = data_params["feature_columns"]
    target_col = data_params["target_column"]
    min_p, max_p = pressure_bounds
    pressures = np.linspace(min_p, max_p, num_points)

    feature_template = np.zeros((1, len(feature_cols)))
    feature_indices = {col: i for i, col in enumerate(feature_cols)}

    if "Temperature (K)" in feature_indices:
        feature_template[0, feature_indices["Temperature (K)"]] = temperature
    else:
        logging.warning(
            "Temperature (K) not in feature columns. Cannot set for pressure evaluation."
        )

    for ion, value in ion_moles.items():
        if ion in feature_indices:
            feature_template[0, feature_indices[ion]] = value
        else:
            logging.warning(
                f"Ion {ion} not in feature columns. Cannot set for pressure evaluation."
            )

    scaler = None
    scaler_path = data_params["scaler_path"]
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path} for pressure evaluation.")
    else:
        logging.warning(
            f"Scaler not found at {scaler_path}. Using unscaled data for pressure evaluation."
        )

    input_dim = len(feature_cols)
    model = BlackBoxNN(
        input_dim=input_dim, hidden_dims=model_params["hidden_dims"], output_dim=1
    )

    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        logging.error(
            f"Model file not found at {model_path}. Cannot evaluate pressure sensitivity."
        )
        return None, None
    except Exception as e:
        logging.error(f"Error loading model state_dict from {model_path}: {e}")
        return None, None

    predictions = []
    with torch.no_grad():
        for pressure_val in pressures:
            features_instance = feature_template.copy()
            if "Pressure (MPa)" in feature_indices:
                features_instance[0, feature_indices["Pressure (MPa)"]] = pressure_val
            else:
                logging.warning(
                    "Pressure (MPa) not in feature columns. Cannot vary pressure for evaluation."
                )

            if scaler:
                features_instance_scaled = scaler.transform(features_instance)
            else:
                features_instance_scaled = features_instance

            X_tensor = torch.tensor(features_instance_scaled, dtype=torch.float32)
            pred = model(X_tensor).cpu().numpy().item()
            predictions.append(pred)

    predictions = np.array(predictions)

    smooth_val = calculate_smoothness_metric(pressures, predictions)
    r2_val = r2_score(pressures, predictions)

    plt.figure(figsize=(10, 6))
    plt.plot(
        pressures,
        predictions,
        plot_color,
        linewidth=2,
        label=f"BlackBox (R²: {r2_val:.3f}, smoothness: {smooth_val:.5f})",
    )
    plt.xlabel("Pressure (MPa)")
    plt.ylabel(f'Predicted {data_params["target_column"]}')
    plt.title(f"BlackBox Model: CO2 Solubility vs Pressure at {temperature} K")
    plt.grid(True)

    try:
        exp_df = pd.read_csv(data_params["experimental_data_path"])
        mask = np.isclose(exp_df["Temperature (K)"], temperature, atol=1)
        for ion, val in ion_moles.items():
            if ion in feature_indices:
                mask &= np.isclose(exp_df[ion], val, atol=0.05)
        exp_df = exp_df[mask]
        if not exp_df.empty:
            plt.scatter(
                exp_df["Pressure (MPa)"],
                exp_df[target_col],
                c="k",
                marker="x",
                label="Experimental",
            )
    except Exception as e:
        logging.warning(f"Could not overlay experimental data: {e}")

    plt.legend()

    filename_base = f"blackbox_co2_vs_pressure_{temperature}K"
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
    logging.info(f"Pressure evaluation plot saved to {plot_save_path}")

    csv_save_path = os.path.join(output_dir, f"{filename_base}.csv")
    results_df = pd.DataFrame(
        {
            "Pressure (MPa)": pressures,
            f'Predicted {data_params["target_column"]}': predictions,
        }
    )
    results_df.to_csv(csv_save_path, index=False)
    logging.info(f"Pressure evaluation CSV saved to {csv_save_path}")

    return pressures, predictions


if __name__ == "__main__":
    best_model_file_path = train_blackbox_model()

    if best_model_file_path and os.path.exists(best_model_file_path):
        print("\n--- Evaluating model on test data ---")
        evaluate_trained_blackbox_model(model_path=best_model_file_path, plot=True)

        print("\n--- Evaluating model response to pressure ---")
        example_temperature = 304
        example_pressure_bounds = (0.1, 100.0)
        example_ion_moles = {"Na+": 0.0, "Cl-": 0.0, "Ca+2": 0.0, "SO4-2": 0.0}
        all_ion_keys = [
            col
            for col in data_params_dict["feature_columns"]
            if col not in ["Temperature (K)", "Pressure (MPa)"]
        ]
        full_example_ion_moles = {
            ion: example_ion_moles.get(ion, 0.0) for ion in all_ion_keys
        }

        evaluate_increasing_pressure(
            model_path=best_model_file_path,
            temperature=example_temperature,
            pressure_bounds=example_pressure_bounds,
            ion_moles=full_example_ion_moles,
        )
    else:
        print(
            "Training did not complete successfully or model file not found. Skipping evaluations."
        )
