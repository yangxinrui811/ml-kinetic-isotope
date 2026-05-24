# PD_Manuscript-2026: Tunneling Phase Diagram Project Summary

## Project Information

- **Paper Title**: A Machine-Learning-Derived Tunneling Phase Diagram Illuminates the Multidimensional Nature of the Kinetic Isotope Effect as a Tunneling Indicator
- **Authors**: Xinrui Yang, Zhigang Wang
- **Institution**: Institute of Atomic and Molecular Physics, Jilin University
- **Target Journal**: JACS / Angewandte / Nature Chemistry
- **Status**: ✅ Completed, ready for submission

## Repository Organization

This repository contains all code, data processing scripts, and analysis tools used in the paper.

### Directory Structure

```
ml-kinetic-isotope/
├── README.md                   # Main documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── src/                        # Source code
│   ├── models/                 # 18 ML model implementations
│   ├── optimization/           # Bayesian hyperparameter optimization
│   ├── analysis/               # Feature importance and SHAP analysis
│   └── visualization/          # Plotting and figure generation
│
├── data/                       # Dataset directory (user-provided)
├── results/                    # Output results and figures
├── docs/                       # Documentation
│   ├── INSTALLATION.md         # Installation guide
│   ├── USAGE_EXAMPLES.md       # Usage examples
│   ├── DATA_FORMAT.md          # Data format specification
│   ├── CONTRIBUTING.md         # Contribution guidelines
│   └── STRUCTURE.md            # Repository structure
│
└── tests/                      # Unit tests
```

## Key Components

### 1. Machine Learning Models (src/models/)

| # | Model | File | Key Feature |
|---|-------|------|-------------|
| 01 | PLSR | `01.PLSR.py` | Partial Least Squares Regression |
| 02 | Ridge | `02.RidgeP-cross.py` | Ridge Regression with cross-validation |
| 03 | ExtraTrees | `03.ExtraTrees-cross.py` | **Best Model (R² = 0.9709)** |
| 04 | DecisionTree | `04.DecisionTree-cross.py` | Decision Tree |
| 05 | RandomForest | `05.RandomForest-cross.py` | Random Forest |
| 06 | AdaBoostDT | `06.AdaBoostDT-cross.py` | AdaBoost with Decision Trees |
| 09 | CatBoost | `09.CatBoost-cross.py` | CatBoost Gradient Boosting |
| - | XGBoost | `XGBoost-cross.py` | XGBoost |
| - | LightGBM | `LightGBM-cross.py` | LightGBM |
| - | GradientBoosting | `GradientBoostingDecisionTree-cross.py` | Gradient Boosting |
| - | SVR | `SVR-cross.py` | Support Vector Regression |
| - | FNN | `FNN-cross.py` | Feedforward Neural Network |
| - | AdaBoostFNN | `AdaBoostFNN-cross.py` | AdaBoost with FNN |
| - | BaggingDT | `BaggingDT-cross.py` | Bagging with Decision Trees |
| - | BaggingFNN | `BaggingFNN-cross.py` | Bagging with FNN |

### 2. Hyperparameter Optimization (src/optimization/BayesSearch/)

- Bayesian optimization using scikit-optimize
- Leave-one-reaction-out cross-validation
- Williams plot analysis for model validation
- Feature importance analysis (SHAP beeswarm plots)
- Prediction analysis at different temperatures (100K, 200K, 300K)

### 3. Key Results

- **Best Model**: ExtraTrees (R² = 0.9709, RMSE = 0.21)
- **Key Features**: Temperature (1/T), KIE, quantum reaction rate
- **Phase Diagram**: Reveals "high KIE - low κ" anomalous regime (300-600 K)
- **Dataset**: 4 amino acids (Ala, Ile, Val, Glu), 20 reaction pathways, 50-1000 K

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ml-kinetic-isotope.git
cd ml-kinetic-isotope

# Install dependencies
pip install -r requirements.txt

# Verify installation
python tests/test_imports.py
```

## Usage

### Quick Start

```bash
# Train best model
python src/models/03.ExtraTrees-cross.py

# Run hyperparameter optimization
python src/optimization/BayesSearch/03.ET.py

# Generate SHAP analysis
python src/optimization/BayesSearch/beeswarm/03.ET.py
```

### Full Pipeline

```bash
# Run all models
for model in src/models/*.py; do
    python "$model"
done

# Run optimization
for opt in src/optimization/BayesSearch/0*.py; do
    python "$opt"
done
```

## Data Requirements

Input data should be CSV format with columns:
- `temperature`: Temperature in Kelvin
- `KIE`: Kinetic Isotope Effect
- `k_cla`: Classical reaction rate (s⁻¹)
- `k_tun`: Quantum tunneling rate (s⁻¹)
- `kappa`: Tunneling correction factor
- `barrier_height`, `barrier_width`, `asymmetry`: PES parameters
- `reaction_type`: Reaction identifier

See `docs/DATA_FORMAT.md` for complete specification.

## Citation

```bibtex
@article{yang2024tunneling,
  title={A Machine-Learning-Derived Tunneling Phase Diagram Illuminates 
         the Multidimensional Nature of the Kinetic Isotope Effect 
         as a Tunneling Indicator},
  author={Yang, Xinrui and Wang, Zhigang},
  journal={J. Am. Chem. Soc.},
  year={2024}
}
```

## Contact

- Xinrui Yang: [EMAIL]
- Zhigang Wang: [EMAIL]
- Institute of Atomic and Molecular Physics, Jilin University

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This work was supported by the Key Laboratory of Material Simulation Methods 
& Software of Ministry of Education, College of Physics, Jilin University.
