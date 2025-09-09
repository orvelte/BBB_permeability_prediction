# Molecular Descriptors

This document describes the molecular descriptors used in the BBB prediction project.

## Overview

Molecular descriptors are numerical representations of chemical structures that capture various physicochemical and structural properties. The RDKit library provides access to over 200 molecular descriptors.

## Categories of Descriptors

### 1. Constitutional Descriptors
- **Molecular Weight (MolWt)**: Total molecular mass
- **Number of Atoms**: Count of all atoms in the molecule
- **Number of Heavy Atoms**: Count of non-hydrogen atoms
- **Number of Rings**: Count of ring systems

### 2. Topological Descriptors
- **Balaban J**: Topological index based on distance matrix
- **Bertz CT**: Complexity index
- **Chi0v, Chi1v, Chi2v**: Valence connectivity indices
- **Kappa1, Kappa2, Kappa3**: Shape descriptors

### 3. Electronic Descriptors
- **Dipole Moment**: Molecular dipole moment
- **Polar Surface Area (TPSA)**: Topological polar surface area
- **Fraction Csp3**: Fraction of sp3 hybridized carbons

### 4. Lipophilicity Descriptors
- **LogP**: Octanol-water partition coefficient
- **LogD**: Distribution coefficient at specific pH
- **Molar Refractivity**: Molecular refractivity

### 5. Pharmacophore Descriptors
- **NumHDonors**: Number of hydrogen bond donors
- **NumHAcceptors**: Number of hydrogen bond acceptors
- **NumRotatableBonds**: Number of rotatable bonds
- **NumAromaticRings**: Number of aromatic rings

## Key Descriptors for BBB Prediction

### Lipinski's Rule of Five
- **MolWt**: < 500 Da
- **LogP**: < 5
- **NumHDonors**: ≤ 5
- **NumHAcceptors**: ≤ 10

### BBB-Specific Descriptors
- **TPSA**: < 90 Å² (good BBB permeability)
- **NumRotatableBonds**: < 10
- **NumAromaticRings**: 0-3

## Descriptor Calculation

```python
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Get all available descriptors
desc_list = [x[0] for x in Descriptors._descList]

# Calculate descriptors for a molecule
mol = Chem.MolFromSmiles('CC(=O)NC1=CC=C(C=C1)O')
calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
descriptors = calc.CalcDescriptors(mol)
```

## Data Preprocessing

### Standardization
- **Z-score normalization**: (x - μ) / σ
- **Min-max scaling**: (x - min) / (max - min)

### Missing Values
- Some descriptors may return NaN for certain molecules
- Handle by removing rows or imputing values

### Feature Selection
- Remove constant features
- Remove highly correlated features
- Use feature importance from tree-based models

## References

1. RDKit Documentation: https://www.rdkit.org/docs/
2. Todeschini, R., & Consonni, V. (2008). Handbook of molecular descriptors.
3. Lipinski, C. A. (2004). Lead- and drug-like compounds: the rule-of-five revolution.
