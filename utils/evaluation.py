import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm

def evaluate_macro_patient_level(df, y_true_col, y_pred_col, group_col='Anon.Patient_ID'):
    grouped = df.groupby(group_col)

    mae_per_patient = grouped.apply(lambda g: mean_absolute_error(g[y_true_col], g[y_pred_col]))
    mse_per_patient = grouped.apply(lambda g: mean_squared_error(g[y_true_col], g[y_pred_col]))
    rmse_per_patient = np.sqrt(mse_per_patient)
    bias_per_patient = grouped.apply(lambda g: np.mean(g[y_pred_col] - g[y_true_col]))

    macro_mae = mae_per_patient.mean()
    macro_mse = mse_per_patient.mean()
    macro_rmse = rmse_per_patient.mean()
    macro_bias = bias_per_patient.mean()

    mape = (
        (np.abs(df[y_pred_col] - df[y_true_col]) / np.maximum(np.abs(df[y_true_col]), 1e-12))
        .mean() * 100
    )
    nrmse = macro_rmse / (df[y_true_col].max() - df[y_true_col].min()) * 100

    summary = {
        'MAE': macro_mae,
        'MSE': macro_mse,
        'RMSE': macro_rmse,
        'Mean Bias Error': macro_bias,
        'MAPE': mape,
        'nRMSE': nrmse,
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
    print(f"Mean Bias Error = {summary['Mean Bias Error']:.3f}")
    print(f"MAPE = {summary['MAPE']:.3f}%")
    print(f"nRMSE = {summary['nRMSE']:.3f}%")



def bland_altman_plots(
    df,
    y_true_col='y_true',
    y_pred_col='y_pred',
    group_col=None,
    abs_title='Bland–Altman: Absolute',
    pct_title='Bland–Altman: Percent Difference',
    x_label="Mean of (Predicted, True)",
    y_abs_label="Predicted − True",
    y_pct_label="Percent Difference (%)",
    sd_limit=1.96,
    figsize=(12, 5),
    show=True,
):
    """
    Plot two Bland-Altman plots: absolute difference and percent difference.
    Optionally averages over group_col (e.g. Patient_ID) if provided.
    """
    # Average over group_col if needed
    if group_col:
        mean_vals = df.groupby(group_col)[[y_true_col, y_pred_col]].mean()
    else:
        mean_vals = df[[y_true_col, y_pred_col]]

    y_true_avg = mean_vals[y_true_col]
    y_pred_avg = mean_vals[y_pred_col]
    diff_avg = y_pred_avg - y_true_avg

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # 1) Absolute Bland-Altman (using statsmodels)
    sm.graphics.mean_diff_plot(
        y_pred_avg, y_true_avg,
        sd_limit=sd_limit,
        scatter_kwds={'alpha': 0.6},
        ax=ax[0]
    )
    ax[0].set_title(abs_title)
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_abs_label)

    # 2) Percent difference Bland-Altman
    mean_pairwise = 0.5 * (y_pred_avg + y_true_avg)
    percent_diff = 100 * diff_avg / mean_pairwise

    bias_pct = np.mean(percent_diff)
    sd_pct = np.std(percent_diff, ddof=1)
    upper_pct = bias_pct + sd_limit * sd_pct
    lower_pct = bias_pct - sd_limit * sd_pct

    ax[1].scatter(mean_pairwise, percent_diff, alpha=0.6)
    ax[1].axhline(bias_pct,  color='gray', linestyle='--')
    ax[1].axhline(upper_pct, color='gray', linestyle='--')
    ax[1].axhline(lower_pct, color='gray', linestyle='--')

    ax[1].set_title(pct_title)
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_pct_label)

    # Annotate bias and LoA
    x_text = ax[1].get_xlim()[0] + 0.05 * (ax[1].get_xlim()[1] - ax[1].get_xlim()[0])
    y_text = ax[1].get_ylim()[1] - 0.05 * (ax[1].get_ylim()[1] - ax[1].get_ylim()[0])
    textstr = (
        f"Bias = {bias_pct:.2f}%\n"
        f"SD   = {sd_pct:.2f}%\n"
        f"95% LoA = [{lower_pct:.2f}%, {upper_pct:.2f}%]"
    )
    ax[1].text(
        x_text, y_text, textstr,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax  

import numpy as np
import matplotlib.pyplot as plt

def bland_altman_pct_comparison(
    test_df,
    val_df,
    y_true_col='y_true',
    y_pred_col='y_pred',
    group_col=None,
    title_test='Test Set',
    title_val='Validation Set',
    x_label="Mean of (Predicted, True)",
    y_label="Percent Difference (%)",
    sd_limit=1.96,
    figsize=(12, 5),
    show=True,
):
    """
    Plot side-by-side percentage Bland-Altman plots for test_df and val_df.
    Optionally average over group_col (e.g. Patient_ID).
    """
    
    def compute_plot_data(df):
        if group_col:
            df = df.groupby(group_col)[[y_true_col, y_pred_col]].mean()
        y_true = df[y_true_col]
        y_pred = df[y_pred_col]
        mean_val = 0.5 * (y_true + y_pred)
        diff_pct = 100 * (y_pred - y_true) / mean_val
        bias = np.mean(diff_pct)
        sd = np.std(diff_pct, ddof=1)
        loa_upper = bias + sd_limit * sd
        loa_lower = bias - sd_limit * sd
        return mean_val, diff_pct, bias, sd, loa_upper, loa_lower

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, df, title in zip(axs, [test_df, val_df], [title_test, title_val]):
        mean_val, diff_pct, bias, sd, upper, lower = compute_plot_data(df)
        ax.scatter(mean_val, diff_pct, alpha=0.6)
        ax.axhline(bias, color='gray', linestyle='--')
        ax.axhline(upper, color='gray', linestyle='--')
        ax.axhline(lower, color='gray', linestyle='--')

        ax.set_title(f'Bland-Altman {title}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Text annotation
        x_text = ax.get_xlim()[0] + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        y_text = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        text = f"Bias = {bias:.2f}%\nSD = {sd:.2f}%\n95% LoA = [{lower:.2f}%, {upper:.2f}%]"
        ax.text(x_text, y_text, text, fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    if show:
        plt.show()
    return fig, axs


def extract_eval_metrics(summary):
    return {
        'MAE': round(summary['MAE'], 4),
        'MSE': round(summary['MSE'], 4),
        'RMSE': round(summary['RMSE'], 4),
        'Bias': round(summary['Mean Bias Error'], 4),
        'MAPE': round(summary['MAPE'], 4),
        'nRMSE': round(summary['nRMSE'], 4),
        'MAE_std': round(summary['per_patient_MAE'].std(), 4),
        'MSE_std': round(summary['per_patient_MSE'].std(), 4),
        'RMSE_std': round(summary['per_patient_RMSE'].std(), 4),
        'N_patients': len(summary['per_patient_MAE'])
    }

def plot_metrics_with_error(df, title):
    plt.figure(figsize=(10,6))
    plt.errorbar(df['n_points'], df['MAE'], yerr=df['MAE_std'], marker='o', capsize=5, label='MAE ± SD')
    plt.errorbar(df['n_points'], df['RMSE'], yerr=df['RMSE_std'], marker='s', capsize=5, label='RMSE ± SD')
    plt.errorbar(df['n_points'], df['MSE'], yerr=df['MSE_std'], marker='^', capsize=5, label='MSE ± SD')
    plt.xlabel('Number of datapoints per combination (n)')
    plt.ylabel('Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()