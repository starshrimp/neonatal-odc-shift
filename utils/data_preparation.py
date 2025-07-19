import numpy as np
from scipy.interpolate import PchipInterpolator

def get_spo2_to_po2_interpolator(odc):
    return PchipInterpolator(odc['SO2 (%)'], odc['PO2 (kPa)'])

def compute_shift_raw(row, spo2_to_po2):
    Pc = spo2_to_po2([row['SpO2(%)']])[0]
    return row['Insp.O2(kPa)'] - Pc

def add_shift_raw_column(df, spo2_to_po2):
    df['shift_raw'] = df.apply(lambda row: compute_shift_raw(row, spo2_to_po2), axis=1)
    return df
