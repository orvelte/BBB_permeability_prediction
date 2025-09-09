# Blood-Brain Barrier (BBB) Permeability Prediction

A machine learning project that predicts whether drugs can cross the Blood-Brain Barrier (BBB) using molecular descriptors and cheminformatics techniques.

## Overview

The Blood-Brain Barrier (BBB) is a selective barrier that prevents most drugs from reaching the brain. This project uses RDKit to extract molecular descriptors from drug SMILES notation and applies machine learning techniques to predict BBB permeability.

## Features

- **Molecular Analysis**: Extract 200+ molecular descriptors from SMILES notation
- **Similarity Analysis**: Compare molecular similarity using Morgan fingerprints and Tanimoto coefficients
- **Exploratory Data Analysis**: Comprehensive EDA with PCA visualization
- **Machine Learning Ready**: Prepared for classification models to predict BBB permeability

## Project Structure

```
BBB_permeability_prediction/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── LICENSE                  # MIT License
├── data/                    # Data directory
│   ├── sample_data.csv      # Sample dataset
│   └── README.md           # Data documentation
├── src/                     # Source code
│   └── bbb.py              # Main analysis script
├── notebooks/               # Jupyter notebooks
│   └── bbb_analysis.ipynb  # Interactive analysis
├── docs/                    # Documentation
│   └── molecular_descriptors.md
├── results/                 # Output files
│   ├── plots/              # Generated plots
│   └── models/             # Trained models
└── tests/                  # Unit tests
    └── test_bbb.py
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd BBB_permeability_prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your BBB dataset as `data/BBB_datasets.csv` with columns:
   - `SMILES`: Chemical structure in SMILES notation
   - `Class`: BBB permeability class (BBB+ or BBB-)

2. Run the analysis:
```bash
python src/bbb.py
```

### Jupyter Notebook

For interactive analysis, use the Jupyter notebook:
```bash
jupyter notebook notebooks/bbb_analysis.ipynb
```

## Dataset Format

The script expects a CSV file with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| SMILES | Chemical structure in SMILES notation | `CC(=O)NC1=CC=C(C=C1)O` |
| Class | BBB permeability class | `BBB+` or `BBB-` |

## Molecular Descriptors

The script extracts 200+ molecular descriptors including:
- Molecular weight
- LogP (lipophilicity)
- Number of rotatable bonds
- Hydrogen bond donors/acceptors
- Topological descriptors
- And many more...

## Output

The script generates:
- Molecular similarity analysis
- PCA visualization of molecular descriptors
- Statistical summaries of the dataset
- Plots showing drug clustering by BBB permeability

## Example Molecules Analyzed

- **Paracetamol**: `CC(=O)NC1=CC=C(C=C1)O`
- **Caffeine**: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
- **Theophylline**: `CN1C2=C(C(=O)N(C1=O)C)NC=N2`
- **MDMA**: `CC(CC1=CC2=C(C=C1)OCO2)NC`

## Dependencies

- **RDKit**: Cheminformatics toolkit
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **scikit-learn**: Machine learning

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RDKit community for the excellent cheminformatics toolkit
- Original Colab notebook: [BBB Analysis](https://colab.research.google.com/drive/1_GQPuZw-g0EQ_iQDrTdlFZJ8BBIE_7OS)

## Future Enhancements

- [ ] Implement machine learning models (Random Forest, SVM, Neural Networks)
- [ ] Add feature importance analysis
- [ ] Cross-validation and model evaluation
- [ ] Web interface for drug prediction
- [ ] API for batch processing
- [ ] Integration with drug databases

## Troubleshooting

### Common Issues

1. **RDKit installation issues**: Try using conda instead of pip:
   ```bash
   conda install -c conda-forge rdkit
   ```

2. **Missing dataset**: Ensure `BBB_datasets.csv` is in the `data/` directory

3. **Memory issues**: For large datasets, consider processing in batches

### Getting Help

- Check the [Issues](https://github.com/yourusername/BBB_permeability_prediction/issues) page
- Create a new issue with detailed error information
- Include your Python version and operating system

## Citation

If you use this project in your research, please cite:

```bibtex
@software{bbb_prediction,
  title={Blood-Brain Barrier Permeability Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/BBB_permeability_prediction}
}
```
