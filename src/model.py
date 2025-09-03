#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


class HousingModel(object):
    """Class for training and evaluating a housing price prediction model."""
    
    def __init__(self):
        """Initialize the model."""
        self.model = LinearRegression()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = {}
        self.feature_names = None
    
    def split_data(self, X, y, test_size=0.2, random_state=0):
        """
        Split the data into training and testing sets.
        
        Args:
            X (DataFrame): Features
            y (Series): Target
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.feature_names = X.columns
        
        print "Training set shape:", self.X_train.shape
        print "Testing set shape:", self.X_test.shape
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self):
        """
        Train the linear regression model.
        
        Returns:
            object: Trained model
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Call split_data() first.")
            
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def predict(self, X=None):
        """
        Make predictions using the trained model.
        
        Args:
            X (DataFrame): Features to predict on. If None, uses X_test
            
        Returns:
            array: Predicted values
        """
        if X is None:
            X = self.X_test
            
        self.y_pred = self.model.predict(X)
        return self.y_pred
    
    def evaluate(self):
        """
        Evaluate the model performance.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.y_pred is None:
            self.predict()
            
        self.metrics = {
            'mse': mean_squared_error(self.y_test, self.y_pred),
            'mae': mean_absolute_error(self.y_test, self.y_pred),
            'r2': r2_score(self.y_test, self.y_pred)
        }
        
        print "Mean squared error: ", self.metrics['mse']
        print "Mean absolute error: ", self.metrics['mae']
        print "R-squared score: ", self.metrics['r2']
        
        # Print model coefficients
        for feature, coef in zip(self.feature_names, self.model.coef_):
            print("{}: {}".format(feature, coef))
            
        return self.metrics
    
    def plot_predictions(self, save_path=None):
        """
        Create a scatter plot of actual vs predicted values.
        
        Args:
            save_path (str): Path to save the plot. If None, displays the plot.
            
        Returns:
            object: Matplotlib figure
        """
        if self.y_pred is None:
            self.predict()
            
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, color='green')
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Housing Prices')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        return plt.gcf()