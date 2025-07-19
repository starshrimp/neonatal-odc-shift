import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_macro_patient_level(df, y_true_col, y_pred_col, group_col='Anon.Patient_ID'):
    grouped = df.groupby(group_col)
    mae_per_patient = grouped.apply(lambda g: mean_absolute_error(g[y_true_col], g[y_pred_col]))
    mse_per_patient = grouped.apply(lambda g: mean_squared_error(g[y_true_col], g[y_pred_col]))
    rmse_per_patient = np.sqrt(mse_per_patient)
    bias_per_patient = grouped.apply(lambda g: np.mean(g[y_pred_col] - g[y_true_col]))

    summary = {
        'MAE': mae_per_patient.mean(),
        'MSE': mse_per_patient.mean(),
        'RMSE': rmse_per_patient.mean(),
        'Bias': bias_per_patient.mean(),
        'per_patient_MAE': mae_per_patient,
        'per_patient_MSE': mse_per_patient,
        'per_patient_RMSE': rmse_per_patient,
        'per_patient_Bias': bias_per_patient,
    }
    return summary

def print_evaluation(summary):
    print("Macro-averaged per-patient metrics:")
    print(f"MAE  = {summary['MAE']:.3f}")
    print(f"MSE  = {summary['MSE']:.3f}")
    print(f"RMSE = {summary['RMSE']:.3f}")
    print(f"Bias = {summary['Bias']:.3f}")
