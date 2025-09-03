#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import urllib2
import ssl
import pandas as pd


class DataLoader(object):
    """Class for loading and processing the Boston Housing dataset."""
    
    def __init__(self, url=None, local_file=None):
        """
        Initialize the DataLoader with URL and local file path.
        
        Args:
            url (str): URL to download the dataset from
            local_file (str): Path to save the downloaded data
        """
        self.url = url or "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
        self.local_file = local_file or "housing.data"
        self.column_names = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", 
            "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"
        ]
        self.df = None
    
    def download_data(self):
        """Download data from URL and save to local file."""
        try:
            # Create an unverified context for SSL
            context = ssl._create_unverified_context()
            
            # Download the data
            response = urllib2.urlopen(self.url, context=context)
            data = response.read()
            
            # Save the data to a local file
            with open(self.local_file, "w") as f:
                f.write(data)
                
            return True
        except Exception as e:
            print("Error downloading data: {}".format(e))
            return False
    
    def load_data(self):
        """Load data from local file into pandas DataFrame."""
        try:
            # Check if file exists, if not download it
            if not os.path.exists(self.local_file):
                self.download_data()
                
            # Load the data into a pandas DataFrame
            self.df = pd.read_csv(
                self.local_file, 
                delim_whitespace=True, 
                names=self.column_names
            )
            
            return self.df
        except Exception as e:
            print("Error loading data: {}".format(e))
            return None
    
    def get_features_and_target(self, features=None):
        """
        Extract features and target from the dataset.
        
        Args:
            features (list): List of feature column names
            
        Returns:
            tuple: (X, y) where X is features DataFrame and y is target Series
        """
        if self.df is None:
            self.load_data()
            
        if features is None:
            features = ['RM', 'LSTAT', 'PTRATIO']
            
        X = self.df[features]
        y = self.df['MEDV']
        
        return X, y