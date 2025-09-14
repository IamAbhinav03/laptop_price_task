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
    
    def _process_cpu_features(self, df):
        """Extracts brand and tier from the 'cpu' column."""
        df['cpu_brand'] = df['cpu'].apply(lambda x: x.split()[0])
        
        def extract_cpu_tier(cpu_str):
            if "Intel Core i7" in cpu_str or "Intel Core i9" in cpu_str:
                return "Intel Core i7/i9"
            elif "Intel Core i5" in cpu_str:
                return "Intel Core i5"
            elif "Intel Core i3" in cpu_str:
                return "Intel Core i3"
            elif "AMD Ryzen" in cpu_str:
                # Captures Ryzen 3, 5, 7, 9
                return "AMD Ryzen"
            else:
                return "Other" # Catches Celeron, Pentium, AMD A-series etc.
        
        df['cpu_tier'] = df['cpu'].apply(extract_cpu_tier)
        return df
    
    def _process_gpu_features(self, df):
        """Extracts brand from the 'gpu' column."""
        df['gpu_brand'] = df['gpu'].apply(lambda x: x.split()[0])
        # We keep it simple to just the brand (Intel, Nvidia, AMD).
        # GPU series is complex and is likely to be highly correlated with CPU tier.
        return df

    def _process_memory_features(self, df):
        """Parses the 'memory' column into separate storage types."""
        df['memory'] = df['memory'].astype(str).replace('\.0', '', regex=True)
        df["memory"] = df["memory"].str.replace('GB', '')
        df["memory"] = df["memory"].str.replace('TB', '000')

        ssd = df["memory"].apply(lambda x: int(re.search(r'(\d+)\s?SSD', x).group(1)) if re.search(r'(\d+)\s?SSD', x) else 0)
        hdd = df["memory"].apply(lambda x: int(re.search(r'(\d+)\s?HDD', x).group(1)) if re.search(r'(\d+)\s?HDD', x) else 0)
        flash = df["memory"].apply(lambda x: int(re.search(r'(\d+)\s?Flash Storage', x).group(1)) if re.search(r'(\d+)\s?Flash Storage', x) else 0)
        
        df['ssd'] = ssd
        df['hdd'] = hdd
        df['flash_storage'] = flash
        return df
    
    def _process_os_features(self, df):
        """Simplifies the 'opsys' column."""
        def map_os(os_str):
            if 'Windows' in os_str:
                return 'Windows'
            elif 'Mac' in os_str:
                return 'Mac'
            elif 'Linux' in os_str:
                return 'Linux'
            else:
                return 'Other/No OS'
        
        df['os_category'] = df['opsys'].apply(map_os)
        return df
    
    def fit(self, df):
        """Learns scaling parameters and column structure from the training data."""
        # We process the dataframe first to know which columns will exist
        temp_df = self.transform(df, is_fitting=True)
        
        # Identify numeric and categorical columns
        self.numeric_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
        if self.target_col in self.numeric_cols:
            self.numeric_cols.remove(self.target_col)
            
        self.categorical_cols = temp_df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Learn scaling parameters
        self.scaling_params['means'] = temp_df[self.numeric_cols].mean()
        self.scaling_params['stds'] = temp_df[self.numeric_cols].std()
        
        # Store the columns from the fitted data (after one-hot encoding)
        self.fitted_columns = temp_df.columns.tolist()
        if self.target_col in self.fitted_columns:
             self.fitted_columns.remove(self.target_col)
        
        return self