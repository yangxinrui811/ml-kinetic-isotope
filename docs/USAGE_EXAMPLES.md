# Usage Examples

## Quick Start

### 1. Training a Single Model

```python
# src/models/03.ExtraTrees-cross.py
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data/processed/reaction_data.csv')

# Features and target
X = data[['KIE', 'temperature', 'k_tun', 'asymmetry']]
y = data['kappa']

# Train model
model = ExtraTreesRegressor(n_estimators=200, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 2. Leave-One-Reaction-Out Cross-Validation

```python
# Custom LOOCV for reaction systems
from sklearn.model_selection import LeaveOneGroupOut

groups = data['reaction_type']
logo = LeaveOneGroupOut()

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    print(f"Left out: {groups.iloc[test_idx].iloc[0]}, R²: {r2:.4f}")
```

### 3. Feature Importance Analysis

```python
import shap

# Train model on full data
model.fit(X, y)

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Summary plot
shap.summary_plot(shap_values, X, plot_type="bar")

# Beeswarm plot
shap.summary_plot(shap_values, X)
```

### 4. Hyperparameter Optimization

```python
# src/optimization/BayesSearch/03.ET.py
from skopt import BayesSearchCV
from skopt.space import Real, Integer

search_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10)
}

opt = BayesSearchCV(
    ExtraTreesRegressor(random_state=42),
    search_space,
    n_iter=50,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

opt.fit(X, y)
print(f"Best parameters: {opt.best_params_}")
print(f"Best R²: {opt.best_score_:.4f}")
```

### 5. Generating the Phase Diagram

```python
# src/visualization/generate_phase_diagram.py
import matplotlib.pyplot as plt
import seaborn as sns

# Predict κ for all data points
y_pred = model.predict(X)

# Create phase diagram
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    data['KIE'], 
    data['kappa'],
    c=data['temperature'],
    cmap='viridis',
    alpha=0.6,
    s=50
)

plt.colorbar(scatter, label='Temperature (K)')
plt.xlabel('Kinetic Isotope Effect (KIE)')
plt.ylabel('Tunneling Correction Factor (κ)')
plt.title('Tunneling Phase Diagram')
plt.xscale('log')
plt.yscale('log')

# Add phase boundaries
plt.axhline(y=2.0, color='r', linestyle='--', label='Strong tunneling (κ=2)')
plt.axhline(y=1.5, color='orange', linestyle='--', label='Moderate tunneling (κ=1.5)')

plt.legend()
plt.tight_layout()
plt.savefig('results/phase_diagram.png', dpi=300)
```

## Advanced Usage

### Custom Model Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_regression, k=10)),
    ('model', ExtraTreesRegressor(n_estimators=200, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Ensemble Model Comparison

```python
from sklearn.ensemble import (
    ExtraTreesRegressor, RandomForestRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

models = {
    'ExtraTrees': ExtraTreesRegressor(n_estimators=200),
    'RandomForest': RandomForestRegressor(n_estimators=200),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200),
    'AdaBoost': AdaBoostRegressor(n_estimators=200),
    'Ridge': Ridge(),
    'PLSR': PLSRegression()
}

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    results[name] = {
        'mean_r2': scores.mean(),
        'std_r2': scores.std()
    }

# Compare results
for name, result in results.items():
    print(f"{name}: R² = {result['mean_r2']:.4f} ± {result['std_r2']:.4f}")
```

## Batch Processing

### Running All Models

```bash
#!/bin/bash
# run_all_models.sh

MODELS=(
    "01.PLSR.py"
    "02.RidgeP-cross.py"
    "03.ExtraTrees-cross.py"
    "04.DecisionTree-cross.py"
    "05.RandomForest-cross.py"
    "06.AdaBoostDT-cross.py"
    "09.CatBoost-cross.py"
)

for model in "${MODELS[@]}"; do
    echo "Running $model..."
    python "src/models/$model"
done
```

## Tips and Best Practices

1. **Memory Management**: For large datasets, use `joblib` for parallel processing
2. **Reproducibility**: Always set `random_state` for reproducible results
3. **Feature Scaling**: Tree-based models don't require scaling, but linear models do
4. **Validation Strategy**: Use stratified sampling for imbalanced reaction types
5. **Hyperparameter Tuning**: Start with coarse search, then fine-tune

## Troubleshooting

### Common Errors

**Error**: `ValueError: could not convert string to float`
- **Solution**: Check data types, ensure categorical variables are encoded

**Error**: `MemoryError`
- **Solution**: Reduce `n_estimators` or use `max_samples` parameter

**Error**: `ImportError: No module named 'shap'`
- **Solution**: Install SHAP: `pip install shap`

## Support

For more examples, see:
- `examples/basic_usage.ipynb` - Jupyter notebook with basic examples
- `examples/advanced_analysis.ipynb` - Advanced analysis techniques
- `docs/API.md` - Complete API documentation
