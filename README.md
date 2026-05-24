# Tunneling Phase Diagram - Machine Learning Analysis

A machine learning framework for decoding the relationship between Kinetic Isotope Effect (KIE) and quantum tunneling correction factor (κ), enabling a predictive tunneling phase diagram.

## Project Overview

This repository contains the computational pipeline for the paper:
> "A Machine-Learning-Derived Tunneling Phase Diagram Illuminates the Multidimensional Nature of the Kinetic Isotope Effect as a Tunneling Indicator"

## Repository Structure

```
.
├── src/                          # Source code
│   ├── models/                   # ML model implementations
│   │   ├── 01.PLSR.py            # Partial Least Squares Regression
│   │   ├── 02.RidgeP-cross.py    # Ridge Regression with cross-validation
│   │   ├── 03.ExtraTrees-cross.py # Extra Trees Regressor
│   │   ├── 04.DecisionTree-cross.py # Decision Tree
│   │   ├── 05.RandomForest-cross.py # Random Forest
│   │   ├── 06.AdaBoostDT-cross.py   # AdaBoost with Decision Trees
│   │   ├── 09.CatBoost-cross.py     # CatBoost
│   │   ├── XGBoost-cross.py         # XGBoost
│   │   ├── LightGBM-cross.py        # LightGBM
│   │   ├── GradientBoostingDecisionTree-cross.py # Gradient Boosting
│   │   ├── SVR-cross.py             # Support Vector Regression
│   │   ├── FNN-cross.py             # Feedforward Neural Network
│   │   ├── AdaBoostFNN-cross.py     # AdaBoost with FNN
│   │   ├── BaggingDT-cross.py       # Bagging with Decision Trees
│   │   └── BaggingFNN-cross.py      # Bagging with FNN
│   ├── optimization/              # Hyperparameter optimization
│   │   └── BayesSearch/            # Bayesian optimization scripts
│   ├── analysis/                   # Analysis scripts
│   │   ├── feature_importance/     # Feature importance analysis
│   │   ├── beeswarm/               # SHAP beeswarm plots
│   │   └── predictions/            # Prediction analysis
│   └── visualization/              # Visualization scripts
│       └── DT-plot/                # Decision tree visualization
├── data/                           # Dataset and processed data
├── results/                        # Output results and figures
├── notebooks/                      # Jupyter notebooks (if any)
├── tests/                          # Unit tests
├── docs/                           # Documentation
├── requirements.txt                # Python dependencies
├── LICENSE                         # License file
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.7+
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- matplotlib
- pandas
- numpy

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-kinetic-isotope.git
cd ml-kinetic-isotope

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Place your reaction rate data in the `data/` directory. The dataset should include:
- Temperature (K)
- Kinetic Isotope Effect (KIE)
- Classical reaction rate (k_cla)
- Quantum reaction rate (k_tun)
- Barrier height (V)
- Barrier width (x2-x1)
- Asymmetry parameter (η)

### 2. Model Training

Run individual models:

```bash
python src/models/03.ExtraTrees-cross.py
```

Or run the complete pipeline:

```bash
python src/run_pipeline.py
```

### 3. Hyperparameter Optimization

```bash
python src/optimization/BayesSearch/03.ET.py
```

### 4. Feature Importance Analysis

```bash
python src/analysis/feature_importance/03.ET.py
```

### 5. Visualization

```bash
python src/visualization/generate_figures.py
```

## Key Results

- **Best Model**: ExtraTrees (R² = 0.9709, RMSE = 0.21)
- **Key Features**: Temperature (1/T), KIE, quantum reaction rate
- **Phase Diagram**: Reveals "high KIE - low κ" anomalous regime (300-600 K)

## Data Availability

The complete dataset and trained model weights are available upon request.

## Citation

If you use this code, please cite:

```bibtex
@article{yang2024tunneling,
  title={A Machine-Learning-Derived Tunneling Phase Diagram Illuminates the Multidimensional Nature of the Kinetic Isotope Effect as a Tunneling Indicator},
  author={Yang, Xinrui and Wang, Zhigang},
  journal={J. Am. Chem. Soc.},
  year={2024}
}
```

## License

MIT License

## Contact

- Xinrui Yang: [EMAIL]
- Zhigang Wang: [EMAIL]
- Institute of Atomic and Molecular Physics, Jilin University
