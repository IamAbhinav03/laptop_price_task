import pandas as pd
import numpy as np
import os
import re

class Preprocessor:
    """
    Handles all feature engineering and preprocessing for the Laptop Price dataset.
    """
    def __init__(self):
        self.target_col = 'price'
        self.numeric_cols = None
        self.categorical_cols = None
        self.fitted_columns = None
        self.scaling_params = {}

    def _clean_and_extract_base_features(self, df):
        """Initial cleaning and extraction of simple numeric features."""
        df.columns = df.columns.str.lower()

        # Drop text from Ram ("8GB" -> 8)
        df['ram_gb'] = df['ram'].str.replace('GB', '').astype(int)

        # Drop text from Weight ("2.5kg" -> 2.5)
        df['weight_kg'] = df['weight'].str.replace('kg', '').astype(float)
        
        return df
    
    def _process_screen_features(self, df):
        """Extracts features from the 'screenresolution' column."""
        df['touchscreen'] = df['screenresolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
        df['ips_panel'] = df['screenresolution'].apply(lambda x: 1 if 'IPS' in x else 0)
        
        # Extract resolution
        resolution = df['screenresolution'].str.extract(r'(\d+)x(\d+)')
        df['x_res'] = resolution[0].astype(int)
        df['y_res'] = resolution[1].astype(int)
        
        # Calculate Pixels Per Inch (PPI)
        df['ppi'] = (((df['x_res']**2) + (df['y_res']**2))**0.5 / df['inches']).astype(float)
        return df