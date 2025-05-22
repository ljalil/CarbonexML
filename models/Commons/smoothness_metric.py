import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_smoothness_metric(x_vals, y_vals):
    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)

    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals must have the same length.")
    
    if len(x_vals) < 3:
        return 0.0 

    sort_indices = np.argsort(x_vals)
    x_vals_sorted = x_vals[sort_indices]
    y_vals_sorted = y_vals[sort_indices]

    d1 = np.gradient(y_vals_sorted, x_vals_sorted, edge_order=1) 
    d2 = np.gradient(d1, x_vals_sorted, edge_order=1)
    
    metric = np.mean(np.abs(d2))
    return metric