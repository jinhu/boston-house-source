#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
import pandas as pd
from src.data_loader import DataLoader

class TestDataLoader(object):
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance for testing."""
        return DataLoader(local_file="test_housing.data")
    
    def test_download_data(self, data_loader):
        """Test that data can be downloaded."""
        result = data_loader.download_data()
        assert result is True
        assert os.path.exists("test_housing.data")
        
        # Clean up
        if os.path.exists("test_housing.data"):
            os.remove("test_housing.data")
    
    def test_load_data(self, data_loader):
        """Test that data can be loaded into a DataFrame."""
        # Create a small test file
        with open("test_housing.data", "w") as f:
            f.write("0.00632 18.00 2.310 0 0.5380 6.575 65.2 4.0900 1 296.0 15.3 396.90 4.98 24.0\n")
            f.write("0.02731 0.00 7.070 0 0.4690 6.421 78.9 4.9671 2 242.0 17.8 396.90 9.14 21.6\n")
        
        df = data_loader.load_data()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 14)
        assert list(df.columns) == data_loader.column_names
        
        # Clean up
        if os.path.exists("test_housing.data"):
            os.remove("test_housing.data")
    
    def test_get_features_and_target(self, data_loader):
        """Test extracting features and target from the dataset."""
        # Create a small test file
        with open("test_housing.data", "w") as f:
            f.write("0.00632 18.00 2.310 0 0.5380 6.575 65.2 4.0900 1 296.0 15.3 396.90 4.98 24.0\n")
            f.write("0.02731 0.00 7.070 0 0.4690 6.421 78.9 4.9671 2 242.0 17.8 396.90 9.14 21.6\n")
        
        X, y = data_loader.get_features_and_target()
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert X.shape == (2, 3)
        assert y.shape == (2,)
        assert list(X.columns) == ['RM', 'LSTAT', 'PTRATIO']
        
        # Test with custom features
        X, y = data_loader.get_features_and_target(['CRIM', 'NOX'])
        assert list(X.columns) == ['CRIM', 'NOX']
        
        # Clean up
        if os.path.exists("test_housing.data"):
            os.remove("test_housing.data")