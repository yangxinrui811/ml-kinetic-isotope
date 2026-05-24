# Installation Guide

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ml-kinetic-isotope.git
cd ml-kinetic-isotope
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n tunneling python=3.8
conda activate tunneling
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python tests/test_imports.py
```

## Data Setup

1. Place your reaction rate data in the `data/` directory
2. Ensure data format matches the expected schema (see `docs/data_format.md`)

## Running the Pipeline

### Step 1: Model Training

```bash
# Run individual models
python src/models/03.ExtraTrees-cross.py

# Or run all models
python src/run_all_models.py
```

### Step 2: Hyperparameter Optimization

```bash
python src/optimization/BayesSearch/03.ET.py
```

### Step 3: Feature Importance Analysis

```bash
python src/analysis/feature_importance/03.ET.py
```

### Step 4: Generate Figures

```bash
python src/visualization/generate_figures.py
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Memory Error**: Reduce dataset size or use incremental learning
   ```python
   # In model scripts, adjust batch size
   batch_size = 1000  # Reduce from default
   ```

3. **Plotting Error**: Install additional plotting dependencies
   ```bash
   pip install plotly bokeh
   ```

## System Requirements

- **Minimum**: 8GB RAM, 2 CPU cores
- **Recommended**: 16GB RAM, 4+ CPU cores, GPU for neural networks
- **Storage**: 10GB free space for datasets and results

## Support

For issues and questions, please open an issue on GitHub or contact:
- Xinrui Yang: [EMAIL]
- Zhigang Wang: [EMAIL]
