# Data Format Specification

## Input Data Format

The dataset should be a CSV file with the following columns:

### Required Columns

| Column Name | Description | Unit | Type |
|------------|-------------|------|------|
| `temperature` | Reaction temperature | K | float |
| `KIE` | Kinetic Isotope Effect | dimensionless | float |
| `k_cla` | Classical reaction rate constant | s⁻¹ | float |
| `k_tun` | Quantum tunneling-corrected rate constant | s⁻¹ | float |
| `kappa` | Tunneling correction factor (κ = k_tun / k_cla) | dimensionless | float |
| `barrier_height` | Potential energy barrier height | kJ/mol | float |
| `barrier_width` | Barrier width parameter | Å | float |
| `asymmetry` | Barrier asymmetry parameter (η) | dimensionless | float |
| `reaction_type` | Reaction identifier (e.g., 'Ala_COOH_a') | string | categorical |

### Optional Columns

| Column Name | Description | Unit | Type |
|------------|-------------|------|------|
| `isotope` | Isotope type ('H' or 'D') | string | categorical |
| `method` | Calculation method ('CVT', 'CVT/SCT') | string | categorical |
| `temperature_range` | Temperature range category | string | categorical |

## Example Data

```csv
temperature,KIE,k_cla,k_tun,kappa,barrier_height,barrier_width,asymmetry,reaction_type
300,15.2,1.23e-10,1.87e-09,15.2,112.5,0.85,0.42,Ala_COOH_a
400,12.8,4.56e-08,5.83e-07,12.8,112.5,0.85,0.42,Ala_COOH_a
500,10.5,3.21e-06,3.37e-05,10.5,112.5,0.85,0.42,Ala_COOH_a
```

## Data Preprocessing

### 1. Data Enhancement

The code supports data enhancement via Arrhenius fitting:

```python
# Original data: 20 points per reaction (50-1000K, 50K intervals)
# Enhanced data: ~1000 points per reaction (1K intervals)
```

### 2. Feature Engineering

Additional features are automatically generated:
- `log_k_cla`: Logarithm of classical rate
- `log_k_tun`: Logarithm of quantum rate
- `inv_temperature`: 1/T (inverse temperature)
- `log_KIE`: Logarithm of KIE

### 3. Data Splitting

- **Training set**: 90% of data (stratified by reaction type)
- **Validation set**: 10% of data
- **Test set**: Leave-one-reaction-out (LOOCV)

## Output Data Format

### Model Predictions

```csv
reaction_type,temperature,actual_kappa,predicted_kappa,residual,model_name
Ala_COOH_a,300,15.2,14.8,0.4,ExtraTrees
```

### Feature Importance

```csv
feature,importance,model_name
KIE,0.35,ExtraTrees
temperature,0.28,ExtraTrees
k_tun,0.22,ExtraTrees
```

## Data Storage

- Raw data: `data/raw/`
- Processed data: `data/processed/`
- Enhanced data: `data/enhanced/`
- Results: `results/`

## Notes

- All rate constants should be in consistent units (s⁻¹)
- Temperature must be in Kelvin
- Missing values should be handled before model training
- Ensure data types match expected formats to avoid runtime errors
