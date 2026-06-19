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

## Step 2: Exploratory Data Analysis (EDA)

Before modeling, we analyze the spatial and temporal properties of the Lorenz 96 system using the `01_data_eda.ipynb` notebook.

* **Trajectory Visualization:** Plots of individual variables over time demonstrate the highly nonlinear, chaotic fluctuations characteristic of the system.
* **Correlation Matrix:** A spatial correlation heatmap reveals strong local dependencies. Variables interact heavily with their immediate neighbors due to the system's differential coupling, while spatially distant variables exhibit near-zero linear correlation.

---

## Step 3: Feature Engineering

To frame the forecasting task as a supervised learning problem, we use a sliding window approach implemented in `src/prepare_windows.py`.

* **Lookback Window:** We use a window of L=10 past time steps to predict the single next time step.
* **Tensor Shapes:** The raw time series is transformed into input tensors of shape `(samples, 10, 40)` and target vectors of shape `(samples, 40)`.

---

## Step 4: Baseline Models

We establish simple baselines in `src/baselines.py` to benchmark our upcoming advanced models. Performance is evaluated using the average Mean Squared Error (MSE) and Mean Absolute Error (MAE) across all 40 state variables.

* **Persistence Baseline:** Assumes the system state remains completely unchanged from the most recently observed time step. 
* **Ridge Regression Baseline:** A linear model that flattens the 400 input features (10 steps * 40 variables) to predict the next 40 target states, utilizing L2 regularization to prevent overfitting.
