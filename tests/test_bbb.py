"""
Unit tests for BBB prediction project
"""

import unittest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ml_models import BBBPredictor

class TestBBBAnalysis(unittest.TestCase):
    """Test cases for BBB analysis functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_smiles = [
            'CC(=O)NC1=CC=C(C=C1)O',  # Paracetamol
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
            'CC(CC1=CC2=C(C=C1)OCO2)NC'  # MDMA
        ]
        
    def test_smiles_parsing(self):
        """Test that SMILES strings can be parsed correctly"""
        for smile in self.sample_smiles:
            mol = Chem.MolFromSmiles(smile)
            self.assertIsNotNone(mol, f"Failed to parse SMILES: {smile}")
            
    def test_molecular_descriptors(self):
        """Test molecular descriptor calculation"""
        mol = Chem.MolFromSmiles(self.sample_smiles[0])
        self.assertIsNotNone(mol)
        
        # Test basic descriptors
        mw = Descriptors.MolWt(mol)
        self.assertGreater(mw, 0)
        
        num_atoms = mol.GetNumAtoms()
        self.assertGreater(num_atoms, 0)
        
    def test_dataframe_creation(self):
        """Test DataFrame creation with sample data"""
        data = {
            'SMILES': self.sample_smiles,
            'Class': ['BBB+', 'BBB+', 'BBB+']
        }
        df = pd.DataFrame(data)
        
        self.assertEqual(len(df), 3)
        self.assertIn('SMILES', df.columns)
        self.assertIn('Class', df.columns)
        
    def test_class_values(self):
        """Test that class values are valid"""
        valid_classes = ['BBB+', 'BBB-']
        test_class = 'BBB+'
        self.assertIn(test_class, valid_classes)
    
    def test_bbb_predictor_initialization(self):
        """Test BBBPredictor initialization"""
        predictor = BBBPredictor(random_state=42)
        self.assertIsNotNone(predictor)
        self.assertEqual(predictor.random_state, 42)
    
    def test_bbb_predictor_data_preparation(self):
        """Test data preparation functionality"""
        # Create sample data
        X = np.random.rand(10, 5)  # 10 samples, 5 features
        y = ['BBB+', 'BBB-'] * 5   # Alternating labels
        
        predictor = BBBPredictor(random_state=42)
        X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
        
        # Check shapes
        self.assertEqual(X_train.shape[0] + X_test.shape[0], 10)
        self.assertEqual(y_train.shape[0] + y_test.shape[0], 10)
        self.assertEqual(X_train.shape[1], 5)
        self.assertEqual(X_test.shape[1], 5)
    
    def test_bbb_predictor_model_training(self):
        """Test model training functionality"""
        # Create sample data
        X = np.random.rand(20, 5)
        y = ['BBB+', 'BBB-'] * 10
        
        predictor = BBBPredictor(random_state=42)
        X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
        
        # Train models
        results = predictor.train_models(X_train, y_train)
        
        # Check that models were trained
        self.assertIn('Random Forest', predictor.trained_models)
        self.assertIn('SVM', predictor.trained_models)
        self.assertIn('Logistic Regression', predictor.trained_models)
        
        # Check results structure
        for model_name in results:
            self.assertIn('cv_mean', results[model_name])
            self.assertIn('cv_std', results[model_name])
    
    def test_bbb_predictor_evaluation(self):
        """Test model evaluation functionality"""
        # Create sample data
        X = np.random.rand(20, 5)
        y = ['BBB+', 'BBB-'] * 10
        
        predictor = BBBPredictor(random_state=42)
        X_train, X_test, y_train, y_test = predictor.prepare_data(X, y)
        
        # Train and evaluate models
        predictor.train_models(X_train, y_train)
        evaluation_results = predictor.evaluate_models(X_test, y_test)
        
        # Check evaluation results
        for model_name in evaluation_results:
            self.assertIn('accuracy', evaluation_results[model_name])
            self.assertIn('predictions', evaluation_results[model_name])
    
    def test_bbb_predictor_feature_importance(self):
        """Test feature importance functionality"""
        # Create sample data
        X = np.random.rand(20, 5)
        y = ['BBB+', 'BBB-'] * 10
        feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        predictor = BBBPredictor(random_state=42)
        X_train, X_test, y_train, y_test = predictor.prepare_data(X, y, feature_names)
        
        # Train models
        predictor.train_models(X_train, y_train)
        
        # Get feature importance
        feature_names_imp, importance_scores = predictor.get_feature_importance(top_n=3)
        
        # Check results
        self.assertEqual(len(feature_names_imp), 3)
        self.assertEqual(len(importance_scores), 3)
        self.assertTrue(all(isinstance(name, str) for name in feature_names_imp))
        self.assertTrue(all(isinstance(score, (int, float)) for score in importance_scores))

if __name__ == '__main__':
    unittest.main()
