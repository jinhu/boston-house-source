#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.data_loader import DataLoader
from src.model import HousingModel

def main():
    """Main function to run the housing price prediction pipeline."""
    # Load data
    loader = DataLoader()
    loader.download_data()
    X, y = loader.get_features_and_target()
    
    # Print data summary
    print "Dataset loaded successfully."
    print "Dataset shape:", loader.df.shape
    print "First few rows:"
    print loader.df.head()
    
    # Create and train model
    model = HousingModel()
    model.split_data(X, y)
    model.train()
    model.predict()
    model.evaluate()
    
    # Plot results
    model.plot_predictions()

if __name__ == "__main__":
    main()