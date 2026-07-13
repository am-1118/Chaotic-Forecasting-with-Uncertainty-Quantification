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

## Step 5: Kolmogorov-Arnold Networks (KAN) & Conformal UQ

We implemented a Time-Series KAN to forecast the chaotic dynamics. Unlike standard MLPs, KANs parameterize non-linear activation functions on the *edges* rather than the nodes.

**Architectural Influences:**
* **[A Practitioner's Guide to Kolmogorov-Arnold Networks](https://arxiv.org/abs/2510.25781):** Guided the formulation of the edge functions as a composite of a base activation (SiLU) and a learnable localized curve (Gaussian RBFs). We applied an L1 sparsity penalty to the weights to encourage localized learning and prune unused edges.
* **[Time Series Forecasting with Hahn KAN](https://arxiv.org/abs/2601.18837):** Guided the macro-architecture for time-series. We bypassed pure channel-independence to preserve the spatial cross-coupling of the Lorenz 96 ring, adopting a bottleneck structure (Input 400 -> Hidden 128 -> Output 40) stabilized by Layer Normalization.

**Performance vs. Baselines (The Reality Check):**
The metrics revealed a critical lesson in dynamical systems forecasting. Because our data was generated with a highly granular time step ($dt=0.01$), the one-step-ahead prediction task is locally smooth. Ridge regression brilliantly exploited this, effectively learning a linear finite-difference approximation of the local gradients to achieve near-perfect short-term accuracy (MSE: 0.0001). 

The KAN fell into the classic deep learning trap of over-parameterization for locally linear tasks. Its highly expressive spline edges memorized the specific chaotic noise of the training set (Train MSE: 0.027) but failed to generalize the continuous physics to unseen data (Test MSE: 1.396). Despite this, the bottleneck architecture proved exceptionally computationally efficient (training in <30 seconds with <25 MB peak VRAM).

**Uncertainty Quantification (Conformal Prediction):**
Because point predictions in chaotic regimes are prone to massive generalization gaps, we implemented **Split Conformal Regression** to mathematically bound the errors.
* We calculated the absolute residuals on the validation set, pooling them across all 40 variables to leverage the translational invariance of the spatial ring.
* We computed a finite-sample corrected quantile.
* Applied to the test set, this distribution-free approach successfully bounded the true chaotic horizon, achieving **89.27% empirical coverage** against our strict theoretical target of 90.0%.

---

## Reproducibility & Tracking

This project enforces strict scientific reproducibility. The raw data generation is hardcoded with a fixed initial perturbation. The training pipelines utilize a global seed (`42`) that locks Python, NumPy, PyTorch, and cuDNN deterministic behaviors, ensuring identical metrics and hardware footprints across runs. 

---

---

## Step 6: Deep Echo State Networks (DeepESN) & Conformal UQ

To bypass the optimization failures and spectral bias of standard continuous neural networks (like PINNs) on long-horizon chaos, we implemented a Deep Echo State Network. 

**Architectural Influences & Methodologies:**
* We built a dynamically sized hierarchical reservoir with leaky-integrator dynamics, explicitly enforcing the Echo State Property at every layer.
* **[Design of Deep Echo State Networks](https://arxiv.org/abs/1912.12423):** We implemented the paper's dynamic architectural design algorithm. By applying a Fast Fourier Transform (FFT) to the state history of each layer, we tracked the shift in the spectral centroid. The algorithm automatically halts the addition of new layers when the filtering effect (low-pass frequency shift) converges.

**Performance vs. Baselines (The Cost of Complexity):**
The FFT algorithm halted the architecture at just 3 layers, proving mathematically that the highly resolved Lorenz 96 system ($dt=0.01$) lacks a deep, multi-timescale hierarchy. 

Furthermore, the DeepESN severely underperformed the simple Ridge Regression baseline (Test MSE: 3.45 vs 0.0001). Because the system's physics are locally linear at $dt=0.01$, the DeepESN took a clean, easily calculable gradient and projected it into a massive, randomized non-linear space, scrambling the signal. This provides empirical proof that injecting deep, randomized non-linearity actively harms one-step-ahead forecasting on highly resolved, fully observable differential equations.

**Uncertainty Quantification:**
Despite the model's poor point predictions, **Split Conformal Regression** successfully bounded the errors, achieving **89.19% empirical coverage** against a strict target of 90.0%. However, to maintain this mathematical guarantee on a poorly generalized model, the algorithm drastically expanded the interval width ($q\_hat \approx 3.00$), perfectly illustrating the trade-off between model accuracy and uncertainty precision.

---

## Conclusion & Summary of Findings

This repository demonstrates a rigorous approach to scientific machine learning on chaotic spatio-temporal systems. By benchmarking advanced architectures against simple baselines, it highlights a critical reality in the field: massive, highly non-linear deep learning models can easily overfit or scramble locally smooth physical dynamics if not applied carefully.

Here is a summary of the architectures implemented and their performance on the Lorenz 96 forecasting task:

* **Ridge Regression Baseline:** Capitalized on the locally linear physics of the highly resolved data (dt=0.01) to achieve near-perfect short-term forecasting. **(Test MSE: 0.0001 | Test MAE: 0.0057)**
* **Kolmogorov-Arnold Networks (KAN):** Implemented using SiLU and Gaussian RBFs on the edges with an L1 sparsity penalty. The highly expressive splines over-parameterized the locally linear task, leading to generalization gaps. **(Test MSE: 1.3965 | Test MAE: 0.9161)**. However, Split Conformal Prediction successfully bounded these errors, achieving **89.27% empirical coverage** with an interval width of **3.69**.
* **Deep Echo State Networks (DeepESN):** Dynamically constructed using an FFT-based stopping algorithm, which halted at 3 layers—proving the high-resolution system lacks a deep multi-timescale hierarchy. The untrained random reservoir scrambled the clean signal, drastically reducing point accuracy. **(Test MSE: 3.4553 | Test MAE: 1.4769)**. Conformal Prediction still maintained mathematical guarantees with **89.19% empirical coverage**, but expanded the interval width to **6.00** to compensate for the high model variance.

Ultimately, the purpose of this repository was to demonstrate a comprehensive understanding of the machine learning for science workflow by implementing, evaluating, and critically analyzing modern research papers.

