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