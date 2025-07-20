import os
import datetime
import pandas as pd
import numpy as np
import json

def log_run_json(identifier, model_type, features, train_subset, test_subset,
                 description, metrics, json_path):
    """
    Appends a structured log entry to a JSON file if not already present.
    """
    import os
    import json

    log_data = []
    run_no = 1

    if os.path.isfile(json_path):
        with open(json_path, "r") as f:
            try:
                log_data = json.load(f)
                if isinstance(log_data, list) and len(log_data) > 0:
                    run_no = max(entry["run"] for entry in log_data) + 1
            except json.JSONDecodeError:
                log_data = []

    # Build current entry (excluding 'run')
    current_entry = {
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

    # Check if an identical entry already exists
    already_logged = any(
        all(entry.get(k) == current_entry.get(k) for k in current_entry)
        for entry in log_data
    )

    if already_logged:
        print("⚠️  Skipped duplicate log entry.")
        return

    # Add run number and log
    current_entry["run"] = run_no
    log_data.append(current_entry)

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"✅ Logged run #{run_no} ➜ {json_path}")

