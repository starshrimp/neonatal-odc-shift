import os
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

def get_root_path():
    # Assume your project root is two levels up from the notebooks
    return Path(os.getcwd()).parents[1]

def load_env(root_path):
    load_dotenv(root_path / ".env")

def load_datasets():
    ROOT = get_root_path()
    load_env(ROOT)
    train_path = Path(ROOT / os.getenv("TRAIN_PATH"))
    test_path  = Path(ROOT / os.getenv("TEST_PATH"))
    val_path   = Path(ROOT / os.getenv("VAL_PATH"))
    odc_path   = Path(ROOT / os.getenv("ODC_PATH"))
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    val_df   = pd.read_csv(val_path)
    odc      = pd.read_csv(odc_path).sort_values('SO2 (%)').drop_duplicates('SO2 (%)')
    return train_df, test_df, val_df, odc
