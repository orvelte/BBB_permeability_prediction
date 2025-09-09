# Data Directory

This directory contains the datasets used for Blood-Brain Barrier (BBB) permeability prediction.

## Required Files

### BBB_datasets.csv
The main dataset containing drug information and BBB permeability labels.

**Expected Format:**
- `SMILES`: Chemical structure in SMILES notation
- `Class`: BBB permeability class (BBB+ for permeable, BBB- for non-permeable)

**Example:**
```csv
SMILES,Class
CC(=O)NC1=CC=C(C=C1)O,BBB+
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,BBB+
CC(CC1=CC2=C(C=C1)OCO2)NC,BBB+
```

## Sample Data

A sample dataset is provided in `sample_data.csv` for testing purposes.

## Data Sources

- Original dataset from cheminformatics research
- SMILES notation from PubChem and other chemical databases
- BBB permeability labels from experimental studies

## Usage

1. Place your `BBB_datasets.csv` file in this directory
2. Ensure the file has the correct column names and format
3. Run the main script from the project root

## Data Privacy

- No personal or proprietary data is included
- All data is publicly available chemical information
- SMILES notation represents chemical structures only

## File Naming Convention

- `BBB_datasets.csv`: Main dataset
- `sample_data.csv`: Sample data for testing
- `*_processed.csv`: Processed/cleaned datasets
- `*_descriptors.csv`: Extracted molecular descriptors
