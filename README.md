# Chaotic Multivariate Forecasting

This repository explores multivariate forecasting of chaotic dynamics, specifically using the Lorenz 96 system. The pipeline covers data generation, exploratory data analysis, simple linear baselines, and advanced methods including Reservoir Computing, Kolmogorov-Arnold Networks (KANs), and SciML with Conformal Uncertainty Quantification.

## Step 1: Data Generation and Baselines

**Dataset:**
- System: Lorenz 96 ($N=40$, $F=8$)
- Integration: RK45, $t \in [0, 250]$, $dt=0.01$
- Snapshots: 20,000 steps after discarding transient dynamics ($t=50$).

**Setup Instructions:**
1. Install requirements: `pip install numpy scipy matplotlib seaborn scikit-learn jupyter`
2. Run data generation: `python src/data_generation.py` (This will populate the `data/` folder).
3. Explore the dataset and evaluate baselines by running the Jupyter notebook: `notebooks/01_data_eda.ipynb`.
