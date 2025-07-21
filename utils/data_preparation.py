import numpy as np
from scipy.interpolate import PchipInterpolator

def get_spo2_to_po2_interpolator(odc):
    return PchipInterpolator(odc['SO2 (%)'], odc['PO2 (kPa)'])

def compute_shift_raw(row, spo2_to_po2):
    Pc = spo2_to_po2([row['SpO2(%)']])[0]
    return row['PiO2(kPa)'] - Pc

def add_shift_raw_column(df, spo2_to_po2):
    df['shift_raw'] = df.apply(lambda row: compute_shift_raw(row, spo2_to_po2), axis=1)
    return df

def add_engineered_features(df, spo2_to_po2):
    df['log_PiO2'] = np.log(df['PiO2(kPa)'])
    df['log_SpO2'] = np.log(df['SpO2(%)'])
    df['SpO2_over_PiO2'] = df['SpO2(%)'] / df['PiO2(kPa)']
    df['SpO2_squared'] = df['SpO2(%)'] ** 2
    df['Hb_SpO2'] = df['Hb'] * df['SpO2(%)']
    df['saturation_deficit'] = 100 - df['SpO2(%)']
    df['CaO2_estimate'] = (1.34 * df['Hb'] * df['SpO2(%)'] / 100) + (0.0031 * df['PiO2(kPa)'])
    df = add_shift_raw_column(df, spo2_to_po2)  # if not already present
    return df