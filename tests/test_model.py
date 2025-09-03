#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np
from src.model import HousingModel

class TestHousingModel(object):
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create synthetic data
        np.random.seed(42)
        X = pd.DataFrame({
            'RM': np.random.normal(6, 1, 100),
            'LSTAT': np.random.normal(12, 5, 100),
            'PTRATIO': np.random.normal(18, 2, 100)
        })
        # Create target with some relationship to features
        y = 3 * X['RM'] - 0.5 * X['LSTAT'] - 0.2 * X['PTRATIO'] + np.random.normal(0, 2, 100)
        y = pd.Series(y)
        
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create a HousingModel instance for testing."""
        return HousingModel()
    
    def test_split_data(self, model, sample_data):
        """Test splitting data into train and test sets."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2)
        
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
    
    def test_train(self, model, sample_data):
        """Test training the model."""
        X, y = sample_data
        model.split_data(X, y)
        trained_model = model.train()
        
        # Check that coefficients are computed
        assert hasattr(trained_model, 'coef_')
        assert len(trained_model.coef_) == 3
    
    def test_predict(self, model, sample_data):
        """Test making predictions."""
        X, y = sample_data
        model.split_data(X, y)
        model.train()
        predictions = model.predict()
        
        assert len(predictions) == 20  # 20% of 100 = 20 test samples
        
        # Test predicting on custom data
        custom_X = pd.DataFrame({
            'RM': [6.0, 7.0],
            'LSTAT': [10.0, 5.0],
            'PTRATIO': [15.0, 16.0]
        })
        custom_pred = model.predict(custom_X)
        assert len(custom_pred) == 2
    
    def test_evaluate(self, model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        model.split_data(X, y)
        model.train()
        model.predict()
        metrics = model.evaluate()
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 0 <= metrics['r2'] <= 1  # R² should be between 0 and 1