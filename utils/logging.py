import os
import datetime
import pandas as pd
import numpy as np
import json

def log_run_json(identifier, model_type, features, train_subset, test_subset,
                 description, metrics, json_path):
    """
    Appends a structured log entry to a JSON file.

    Parameters:
    - identifier (str): Unique ID or label for the run.
    - model_type (str): E.g. 'LinearRegression', 'XGBoost', etc.
    - features (list[str]): List of feature names used in the model.
    - train_subset (str): Short description of training subset used.
    - test_subset (str): Short description of test subset used.
    - description (str): Free-text notes on this run.
    - metrics (dict): Dictionary of scalar metrics.
    - json_path (str): Where to append/store the run logs.
    """
    import os
    import json

    run_no = 1
    log_data = []


    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            try:
                log_data = json.load(f)
                if isinstance(log_data, list) and len(log_data) > 0:
                    run_no = max(entry["run"] for entry in log_data) + 1
            except json.JSONDecodeError:
                log_data = []


    entry = {
        "run": run_no,
        "id": identifier,
        "model_type": model_type,
        "features": features,
        "train_subset": train_subset,
        "test_subset": test_subset,
        "metrics": {
            k: float(v) if isinstance(v, (np.generic, np.floating)) else v
            for k, v in metrics.items()
            if not isinstance(v, pd.Series)
        },
        "description": description.strip(),
    }

    # Save entry
    log_data.append(entry)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"✅ Logged run #{run_no} ➜ {json_path}")
