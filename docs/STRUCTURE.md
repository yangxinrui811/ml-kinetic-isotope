# Repository Structure

```
ml-kinetic-isotope/
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── README.md                   # Main project documentation
├── requirements.txt            # Python dependencies
│
├── data/                       # Data directory (user-provided)
│   ├── raw/                    # Original experimental/computational data
│   ├── processed/              # Cleaned and preprocessed data
│   └── enhanced/               # Data enhanced via Arrhenius fitting
│
├── docs/                       # Documentation
│   ├── CONTRIBUTING.md         # How to contribute to the project
│   ├── DATA_FORMAT.md          # Data format specification
│   ├── INSTALLATION.md         # Installation guide
│   └── USAGE_EXAMPLES.md       # Usage examples and tutorials
│
├── results/                    # Output results (generated)
│   ├── Figure/                 # Generated figures
│   └── decision_tree.png       # Example decision tree visualization
│
├── src/                        # Source code
│   ├── models/                 # Machine learning model implementations
│   │   ├── 01.PLSR.py              # Partial Least Squares Regression
│   │   ├── 02.RidgeP-cross.py    # Ridge Regression with cross-validation
│   │   ├── 03.ExtraTrees-cross.py # Extra Trees Regressor (Best Model)
│   │   ├── 04.DecisionTree-cross.py # Decision Tree
│   │   ├── 05.RandomForest-cross.py # Random Forest
│   │   ├── 06.AdaBoostDT-cross.py   # AdaBoost with Decision Trees
│   │   ├── 09.CatBoost-cross.py     # CatBoost
│   │   ├── AdaBoostFNN-cross.py     # AdaBoost with Neural Network
│   │   ├── BaggingDT-cross.py       # Bagging with Decision Trees
│   │   ├── BaggingFNN-cross.py      # Bagging with Neural Network
│   │   ├── DecisionTree-cross-test.py # Decision Tree test version
│   │   ├── DecisionTree.py          # Simple Decision Tree
│   │   ├── FNN-cross.py             # Feedforward Neural Network with CV
│   │   ├── FNN.py                   # Simple Feedforward Neural Network
│   │   ├── GradientBoostingDecisionTree-cross.py # Gradient Boosting
│   │   ├── LightGBM-cross.py        # LightGBM
│   │   ├── SVR-cross.py             # Support Vector Regression
│   │   └── XGBoost-cross.py         # XGBoost
│   │
│   ├── optimization/           # Hyperparameter optimization
│   │   └── BayesSearch/        # Bayesian optimization scripts
│   │       ├── 01.PLSR.py
│   │       ├── 02.RidgeP.py
│   │       ├── 03.ET.py
│   │       ├── 03.SVR.py
│   │       ├── 04.RF.py
│   │       ├── 05.CB.py
│   │       ├── 06.XB.py
│   │       ├── 07.GD.py
│   │       ├── 08.LG.py
│   │       ├── CatBoost-Bayes.py
│   │       ├── DecisionTree-Bayes.py
│   │       ├── For20-LOO/          # Leave-one-out cross-validation results
│   │       │   ├── CB/
│   │       │   ├── ET/
│   │       │   ├── GB/
│   │       │   ├── LG/
│   │       │   ├── PLSR/
│   │       │   ├── RF/
│   │       │   ├── Ridge/
│   │       │   ├── Williams/       # Williams plot analysis
│   │       │   ├── XB/
│   │       │   └── figures.pptx
│   │       ├── beeswarm/           # SHAP beeswarm plots
│   │       │   ├── 01.PLSR.py
│   │       │   ├── 02.RidgeP.py
│   │       │   ├── 03.ET.py
│   │       │   ├── 04.RF.py
│   │       │   ├── 05.CB.py
│   │       │   ├── 06.XB.py
│   │       │   ├── 07.GD.py
│   │       │   ├── 08.LG.py
│   │       │   └── figures/
│   │       ├── plots/               # Generated figures
│   │       ├── feature_importance/          # Feature importance analysis
│   │       │   ├── 01.PLSR.py
│   │       │   ├── 02.RidgeP.py
│   │       │   ├── 03.ET.py
│   │       │   ├── 04.RF.py
│   │       │   ├── 05.CB.py
│   │       │   ├── 06.XB.py
│   │       │   ├── 07.GD.py
│   │       │   ├── 08.LG.py
│   │       │   └── 重要性plots/
│   │       ├── predictions/               # Prediction scripts
│   │       │   ├── 01.PLSR.py
│   │       │   ├── 02.RidgeP.py
│   │       │   ├── 03.ET.py
│   │       │   ├── 04.RF.py
│   │       │   ├── 05.CB.py
│   │       │   ├── 06.XB.py
│   │       │   ├── 07.GD.py
│   │       │   ├── 08.LG.py
│   │       │   ├── 100Kpredictions_matrix.csv
│   │       │   ├── 200Kpredictions_matrix.csv
│   │       │   ├── 300Kpredictions_matrix.csv
│   │       │   ├── predictions.csv
│   │       │   ├── predictions_matrix.csv
│   │       │   ├── figuresSHAP/
│   │       │   ├── multivariate_analysis/
│   │       │   └── predictions值关系/
│   │       └── error_results.xlsx
│   │
│   ├── analysis/               # Analysis scripts (to be organized)
│   └── visualization/          # Visualization scripts
│       └── DT-plot/            # Decision tree plotting
│
└── tests/                      # Unit tests
    └── test_imports.py         # Import verification test
```

## Key Files

| File | Description |
|------|-------------|
| `README.md` | Main project documentation |
| `requirements.txt` | Python package dependencies |
| `LICENSE` | MIT License |
| `.gitignore` | Git ignore rules |
| `src/models/03.ExtraTrees-cross.py` | Best performing model (R² = 0.9709) |
| `src/optimization/BayesSearch/` | Hyperparameter optimization |
| `docs/` | Complete documentation |

## Notes

- **Data files** are not included in the repository (see `.gitignore`)
- **Results** are generated during execution and stored in `results/`
- **BayesSearch** directory contains extensive optimization and analysis results
- All model scripts follow the naming convention: `[Number].[Algorithm]-[Feature].py`

## Next Steps for Organization

1. **Consolidate analysis scripts** from `BayesSearch/` into `src/analysis/`
2. **Move generated figures** from `BayesSearch/plots/` to `results/figures/`
3. **Organize prediction results** into `results/predictions/`
4. **Create unified pipeline script** in `src/run_pipeline.py`
5. **Add unit tests** for each model in `tests/`
