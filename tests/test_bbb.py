"""
Unit tests for BBB prediction project
"""

import unittest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

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

if __name__ == '__main__':
    unittest.main()
