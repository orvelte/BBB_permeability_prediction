#!/usr/bin/env python3
"""
BBB Permeability Prediction Script

This script allows you to predict BBB permeability for new molecules
using the trained machine learning models.
"""

import sys
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import joblib

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from ml_models import BBBPredictor


def calculate_descriptors(smiles_list):
    """
    Calculate molecular descriptors for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Tuple of (descriptors_array, valid_smiles, feature_names)
    """
    descriptors = []
    valid_smiles = []
    
    # Get descriptor names
    feature_names = [x[0] for x in Descriptors._descList]
    
    for smile in smiles_list:
        mol = Chem.MolFromSmiles(smile)
        if mol is not None:
            try:
                calc = MoleculeDescriptors.MolecularDescriptorCalculator(feature_names)
                vector = calc.CalcDescriptors(mol)
                descriptors.append(vector)
                valid_smiles.append(smile)
            except Exception as e:
                print(f"Error processing SMILES: {smile} - {e}")
        else:
            print(f"Invalid SMILES: {smile}")
    
    return np.array(descriptors), valid_smiles, feature_names


def predict_bbb_permeability(smiles_list, model_dir='../results/models', model_name=None):
    """
    Predict BBB permeability for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        model_dir: Directory containing trained models
        model_name: Name of model to use (if None, uses best model)
        
    Returns:
        DataFrame with predictions
    """
    # Calculate descriptors
    descriptors, valid_smiles, feature_names = calculate_descriptors(smiles_list)
    
    if len(descriptors) == 0:
        print("No valid molecules found.")
        return pd.DataFrame()
    
    # Load the predictor
    predictor = BBBPredictor()
    predictor.load_models(model_dir)
    
    # Make predictions
    predictions = predictor.predict(descriptors, model_name)
    probabilities = predictor.predict_proba(descriptors, model_name)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'SMILES': valid_smiles,
        'Predicted_Class': predictions,
        'BBB+_Probability': probabilities[:, 1],
        'BBB-_Probability': probabilities[:, 0]
    })
    
    return results_df


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python predict_bbb.py <SMILES1> [SMILES2] ...")
        print("Example: python predict_bbb.py 'CC(=O)NC1=CC=C(C=C1)O' 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'")
        sys.exit(1)
    
    # Get SMILES from command line
    smiles_list = sys.argv[1:]
    
    # Make predictions
    results = predict_bbb_permeability(smiles_list)
    
    if not results.empty:
        print("\nBBB Permeability Predictions:")
        print("=" * 50)
        print(results.to_string(index=False))
        
        # Save results
        output_file = 'bbb_predictions.csv'
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("No valid predictions could be made.")


if __name__ == "__main__":
    main()
