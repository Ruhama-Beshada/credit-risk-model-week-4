# File: tests/test_data_processing.py

import unittest
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Example helper function to test (replace with your actual function)
def feature_engineering(df):
    # Just a sample: suppose we create 'RFM_Total' column
    df['RFM_Total'] = df['Recency'] + df['Frequency'] + df['Monetary']
    return df

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Sample data
        self.df = pd.DataFrame({
            'Recency': [10, 20, 5],
            'Frequency': [3, 2, 7],
            'Monetary': [100, 200, 50],
            'Age': [25, np.nan, 35]
        })
        
        # Preprocessing pipeline example (for testing NaN handling)
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
    
    def test_feature_engineering_columns(self):
        """Test if feature engineering adds 'RFM_Total' column"""
        df_transformed = feature_engineering(self.df.copy())
        self.assertIn('RFM_Total', df_transformed.columns)
    
    def test_pipeline_handles_missing_values(self):
        """Test that pipeline can process data with missing values"""
        processed = self.pipeline.fit_transform(self.df[['Age']])
        # Should not contain NaNs after imputation
        self.assertFalse(np.isnan(processed).any())

if __name__ == "__main__":
    unittest.main()
