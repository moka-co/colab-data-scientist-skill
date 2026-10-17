# Essential Data Science, Machine Learning & Computational AI Python Libraries

An authoritative directory of Python libraries categorized by functional domain and task. This guide serves as an on-demand reference for the Main Orchestrator and subagents when configuring Colab environments, recommending dependencies, and planning workflows.

---

## ⚠️ Important Colab Runtime & Environment Operational Notes
> [!IMPORTANT]
> **Pre-installed in Google Colab (Standard CPU/GPU runtimes):**
> - Standard scientific stack: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`.
> - Deep learning standard stack: `torch`, `torchvision`, `torchaudio`, `tensorflow`, `keras`.
>
> **Requires Kernel Restart After Installation:**
> - Any package that upgrades or replaces pre-installed C-extensions/binary wheels (e.g. upgrading `numpy`, `scipy`, `pillow`, or enabling `cudf.pandas`) requires restarting the Colab Python runtime via code or UI:
>   ```python
>   # In Colab cell to restart kernel programmatically:
>   import os
>   os.kill(os.getpid(), 9)
>   ```
>
> **High-Risk Conflict Traps for `env-setup-agent`:**
> - **PyTorch & CUDA mismatch**: Installing packages that pull unpinned `torch` versions can break Colab's pre-installed CUDA acceleration. Always specify `--no-deps` or pin to Colab's active PyTorch/CUDA version.
> - **FlashAttention / DeepSpeed**: Compiling C++/CUDA extensions from source in Colab often fails or takes 15+ minutes. Prefer pre-built wheels matching Colab's exact torch/CUDA version or lightweight alternatives like `unsloth` or `bitsandbytes`.
> - **cuDF / RAPIDS**: Requires a GPU runtime (T4/V100/A100) and must be loaded via `%load_ext cudf.pandas` before importing pandas.

---

## 🧭 Out-of-Pipeline Domains Notice
> [!NOTE]
> Sections marked with **`[Out-of-Pipeline]`** represent specialized domains (e.g., Mathematical Optimization, Formal Verification, Physics Simulation, Distributed Serving, and Federated Learning).
> **Workflow Rule**: When a user's task centers on an `[Out-of-Pipeline]` domain, the Main Orchestrator must **not** force the standard tabular notebook pipeline (EDA $\rightarrow$ Cleaning $\rightarrow$ Model Training). Instead, the Orchestrator must ask the user (`ask_question`) or dispatch a task-appropriate specialized flow.

---

## 📍 Section-to-Line Index Map
The Main Orchestrator MUST use this map to inspect specific line ranges with `view_file` (specifying `StartLine` and `EndLine`) or grep headers, rather than reading the entire document.

| Part | Section # | Domain Name | Out-of-Pipeline? | Line Range |
|---|---|---|---|---|
| **Part I** | §1 | Dataframe Engines & Structured Data | No | L73-L89 |
| | §2 | EDA & Automated Profiling | No | L91-L101 |
| | §3 | Data Visualization & Dashboards | No | L103-L118 |
| | §4 | Data Validation & Orchestration | No | L120-L130 |
| | §5 | Synthetic Data Generation | No | L132-L142 |
| **Part II** | §6 | Classical Machine Learning & AutoML | No | L144-L163 |
| | §7 | Statistical Modeling & Bayesian Inference | No | L165-L178 |
| | §8 | Optimization, SAT, SMT & Planning | **Yes [Out-of-Pipeline]** | L180-L196 |
| | §9 | Scientific Computing & Symbolic Math | No | L198-L210 |
| | §10 | Graph Analytics & Graph Neural Networks | No | L212-L223 |
| **Part III**| §11 | Time Series Forecasting & Anomaly | No | L225-L245 |
| | §12 | Causal Inference & Uplift Modeling | No | L247-L257 |
| | §13 | Survival Analysis & Event History | No | L259-L268 |
| | §14 | Recommender Systems | No | L270-L280 |
| | §15 | Geospatial Analytics & Mapping | No | L282-L293 |
| | §16 | Bioinformatics & Genomics | No | L295-L307 |
| | §17 | Simulation & Physics Engines | **Yes [Out-of-Pipeline]** | L309-L323 |
| **Part IV** | §18 | Deep Learning & LLM Fine-Tuning | No | L325-L343 |
| | §19 | Reinforcement Learning & Simulation | No | L345-L361 |
| | §20 | NLP, Embeddings, RAG & Vector DBs | No | L363-L419 |
| | §21 | Foundation Model SDKs & Multi-Provider | No | L421-L432 |
| | §22 | AI Agents & Multi-Agent Frameworks | No | L434-L450 |
| | §23 | Computer Vision & Generative Media | No | L452-L472 |
| | §24 | Audio, Speech & Video Processing | No | L474-L492 |
| **Part V**  | §25 | XAI, Fairness & Data Drift Monitoring | No | L494-L509 |
| | §26 | Hyperparameter Tuning & Tracking | No | L511-L526 |
| | §27 | Federated Learning & Privacy | **Yes [Out-of-Pipeline]** | L528-L539 |
| | §28 | Model Serving, Interactive UIs & APIs | **Yes [Out-of-Pipeline]** | L541-L549 |

---

# Part I: Core Data Engineering & Exploratory Analytics

## 1. Dataframe Engines & Structured Data Processing
High-performance libraries for handling tabular data, out-of-core computing, GPU dataframes, and SQL database access.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **pandas** | `pandas` | `pandas` | Universal tabular manipulation. De-facto Python standard; pre-installed in Colab; best for `< 2–5 GB` datasets in memory. |
| **Polars** | `polars` | `polars` | Ultra-fast multi-threaded dataframes written in Rust. Lazy query optimization; Apache Arrow memory model; minimal RAM overhead. |
| **DuckDB** | `duckdb` | `both` | In-process analytical SQL OLAP. Zero-copy Arrow integration; excels at querying Parquet/CSV files directly via standard SQL. |
| **Modin** | `modin[ray]` | `pandas` | Drop-in pandas parallelization. Transparently distributes pandas operations across CPU cores using Ray or Dask backends. |
| **Vaex** | `vaex` | `both` | Out-of-core lazy tabular computing. Visualizes and computes over 100M+ rows / huge CSV/HDF5 files with zero-memory memory-mapping. |
| **cuDF (RAPIDS)** | `cudf-cu12` | `pandas` | GPU-accelerated dataframes. Drop-in GPU pandas API (`cudf.pandas`); requires Colab GPU runtime. |
| **PySpark** | `pyspark` | `both` | Distributed big data & cluster computing. Industry standard for terabyte-scale distributed datasets and SQL queries. |
| **Dask** | `dask[dataframe]` | `both` | Parallel out-of-core computing. Scales pandas and NumPy workflows across multiple CPU cores or clusters. |
| **SQLAlchemy** | `sqlalchemy` | `both` | SQL database access & ORM. Standard tool for connecting to relational databases, connection pooling, and schema mapping. |
| **PyArrow** | `pyarrow` | `both` | Columnar in-memory data interchange. Interoperability foundation for Feather, Parquet, and zero-copy cross-engine sharing. |

---

## 2. Exploratory Data Analysis & Automated Profiling
Tools to rapidly diagnose schemas, missingness patterns, and generate automated profiling reports.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **ydata-profiling** | `ydata-profiling` | `pandas` | Automated HTML EDA reports. Generates complete summary reports (correlations, missingness, distributions) with a single command. |
| **missingno** | `missingno` | `pandas` | Missing data visualization. Visual matrices, heatmaps, and dendrograms of missing value distributions. |
| **Sweetviz** | `sweetviz` | `pandas` | Target-oriented EDA & dataset comparison. Generates high-density visual comparison between train and test datasets. |
| **Great Expectations** | `great-expectations` | `both` | Data profiling & contract assertion. Asserts schema integrity and statistical invariants during ingestion. |

---

## 3. Data Visualization & Interactive Dashboards
Static, publication-quality, declarative, and interactive visualization toolkits.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Matplotlib** | `matplotlib` | `both` | Fundamental plotting engine. Base plotting library offering pixel-level control; pre-installed in Colab. |
| **Seaborn** | `seaborn` | `both` | Statistical graphics & themes. Built on Matplotlib; elegant statistical plots (KDE, pairplots, boxenplots, heatmaps). |
| **Plotly** | `plotly` | `both` | Interactive WebGL figures. Pan, zoom, hover tooltips; seamlessly renders inside Google Colab cells. |
| **plotnine** | `plotnine` | `pandas` | Grammar of Graphics (ggplot2 port). Idiomatic implementation of the R ggplot2 grammar of graphics for Python. |
| **HoloViews** | `holoviews` | `both` | High-level declarative plotting over Bokeh, Matplotlib, and Plotly with interactive widgets. |
| **hvPlot** | `hvplot` | `both` | Simple `.hvplot()` plotting API for pandas, Polars, and Xarray using HoloViews. |
| **Pygwalker** | `pygwalker` | `both` | Tableau-like visual notebook exploration. Drag-and-drop visual analytics embedded directly inside notebook cells. |
| **Altair** | `altair` | `both` | Declarative statistical charts based on Vega-Lite grammar of graphics. |
| **Bokeh** | `bokeh` | `both` | High-performance interactive visualizations for large or streaming datasets. |

---

## 4. Data Validation, Testing & Pipeline Orchestration
Data contracts, schema enforcement, and workflow automation.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Pydantic** | `pydantic` | `n/a` | Data validation using Python type hints. Schema contracts, JSON parsing, and runtime input validation. |
| **Pandera** | `pandera` | `both` | Statistical schema validation for dataframes. Expressive schema checks and hypothesis tests for pandas, Polars, and PySpark. |
| **Prefect** | `prefect` | `n/a` | Modern workflow orchestrator. Dynamic task scheduling, retry handling, and Pythonic pipeline lineage. |
| **Dagster** | `dagster` | `n/a` | Asset-based data workflow orchestrator for data engineering and ML data lineage. |

---

## 5. Synthetic Data Generation & Anonymization
Generating realistic test data, privacy-preserving synthetic tables, and mock records.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **SDV** | `sdv` | `pandas` | Statistical synthetic tabular data. Learns from real tables and generates new ones using Gaussian copulas, CTGAN, and TVAE. |
| **Faker** | `faker` | `n/a` | Realistic fake records for testing. Generates localized mock names, addresses, emails, dates, and transactions. |

---

# Part II: Classical Machine Learning, Optimization & Mathematics

## 6. Classical Machine Learning & Feature Engineering
Algorithms for tabular classification, regression, clustering, resampling, and AutoML.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **scikit-learn** | `scikit-learn` | `both` | Standard ML toolkit. Pipelines, estimators, cross-validation, and metrics; pre-installed in Colab. |
| **imbalanced-learn** | `imbalanced-learn` | `pandas` | Resampling for class imbalance. SMOTE, ADASYN, Tomek links, and balanced pipeline wrappers. |
| **Feature-engine** | `feature-engine` | `pandas` | Feature engineering transformers for imputation, discretization, categorical encoding, and outlier capping. |
| **category_encoders** | `category-encoders` | `pandas` | Categorical feature encoding. Target encoding, CatBoost encoding, Weight of Evidence, and hashing encoders. |
| **LightGBM** | `lightgbm` | `both` | Gradient boosted decision trees. Histogram-based, leaf-wise splitting; fast and memory-efficient. |
| **XGBoost** | `xgboost` | `both` | Extreme gradient boosting. High-performance GBDT with built-in regularization and GPU support. |
| **CatBoost** | `catboost` | `both` | Gradient boosting for categorical data. Handles categorical features natively without explicit one-hot encoding. |
| **AutoGluon** | `autogluon` | `both` | Automated Machine Learning (AutoML). Multi-layer stacking and model ensembling across tabular data. |
| **FLAML** | `flaml` | `both` | Fast, lightweight AutoML and hyperparameter tuning with cost-effective search strategies. |
| **PyCaret** | `pycaret` | `pandas` | Low-code machine learning. End-to-end automated workflow covering preprocessing, training, and tuning. |
| **UMAP-learn** | `umap-learn` | `both` | Manifold learning and non-linear dimensionality reduction preserving local and global structures. |
| **openTSNE** | `opentsne` | `both` | Fast, modular t-SNE implementation supporting out-of-core transforms and large datasets. |
| **HDBSCAN** | `hdbscan` | `both` | Hierarchical density-based clustering without requiring prior specification of cluster counts. |

---

## 7. Statistical Modeling, Hypothesis Testing & Bayesian Inference
Econometrics, inferential statistics, Markov models, and probabilistic programming.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **statsmodels** | `statsmodels` | `pandas` | Econometrics & hypothesis testing. OLS, GLM, ANOVA, and detailed statistical summary tables. |
| **Pingouin** | `pingouin` | `pandas` | User-friendly statistical package. Simpler, pandas-oriented API for t-tests, ANOVA, and effect sizes. |
| **PyMC** | `pymc` | `n/a` | Bayesian modeling and probabilistic programming using No-U-Turn Samplers (NUTS). |
| **NumPyro** | `numpyro` | `n/a` | Probabilistic programming on JAX for high-performance GPU-accelerated MCMC inference. |
| **Bambi** | `bambi` | `pandas` | High-level formula-based interface for Bayesian generalized linear models built on PyMC. |
| **pgmpy** | `pgmpy` | `pandas` | Probabilistic Graphical Models (PGMs). Bayesian Networks, exact/approximate inference, and d-separation queries. |
| **pomegranate** | `pomegranate` | `n/a` | GPU-accelerated probabilistic models in PyTorch (mixture models, HMMs, Bayesian classifiers). |

---

## 8. Mathematical Optimization, SAT, SMT & Automated Planning `[Out-of-Pipeline]`
Constraint satisfaction, mixed-integer programming (MILP), satisfiability, and PDDL planning.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **OR-Tools** | `ortools` | `n/a` | Industrial constraint programming, routing & scheduling. Google CP-SAT solver for scheduling and assignment. |
| **python-sat (PySAT)** | `python-sat` | `n/a` | SAT & MaxSAT solving. Unified API to solvers (Glucose, CaDiCaL, MiniSat) with cardinality and RC2 encodings. |
| **Z3** | `z3-solver` | `n/a` | SMT solving and formal verification. Microsoft SMT solver for integers, reals, bit-vectors, and program logic. |
| **PuLP** | `pulp` | `n/a` | Accessible LP / MILP modeling. Clean syntax for linear optimization; bundles CBC solver; interfaces with HiGHS/Gurobi. |
| **Pyomo** | `pyomo` | `n/a` | Full algebraic modeling language (AML) supporting linear, nonlinear, and differential-algebraic equations. |
| **CVXPY** | `cvxpy` | `n/a` | Disciplined Convex Programming (DCP) verifying convexity and routing to conic/semidefinite solvers. |
| **python-constraint** | `python-constraint` | `n/a` | Lightweight CSP backtracking solver for classic constraint puzzles (Sudoku, N-Queens). |
| **Unified Planning** | `unified-planning` | `n/a` | AIPlan4EU framework to model automated planning problems once and execute across multiple PDDL engines. |
| **pyperplan** | `pyperplan` | `n/a` | Educational PDDL planner in pure Python implementing BFS, A*, and heuristic search (hFF, hmax). |
| **pyDatalog** | `pyDatalog` | `n/a` | Logic programming in Python with Datalog-style deductive rules and queries. |

---

## 9. Scientific Computing, Symbolic Math & JIT Acceleration
Core numerics, analytical mathematics, and high-performance compilation.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **NumPy** | `numpy` | `both` | Fundamental N-dimensional array computing underpinning all scientific Python; pre-installed in Colab. |
| **SciPy** | `scipy` | `both` | Numerical integration, linear algebra, optimization, and statistical distributions; pre-installed in Colab. |
| **SymPy** | `sympy` | `n/a` | Symbolic mathematics and computer algebra system for calculus, matrix equations, and analytical solving. |
| **Numba** | `numba` | `both` | LLVM-based JIT compiler that converts NumPy-heavy Python loops into machine-code speeds. |
| **PySR** | `pysr` | `both` | High-performance genetic programming (powered by Julia) to discover analytical symbolic equations from data. |
| **Taichi** | `taichi` | `n/a` | Compiles Python code to GPU/CPU parallel kernels for physically-based simulation and graphics. |

---

## 10. Graph Analytics & Graph Neural Networks
Complex networks, structural metrics, and deep learning on graphs.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **NetworkX** | `networkx` | `both` | Graph theory and network analysis. Pathfinding, centrality, community detection, and graph algorithms. |
| **PyTorch Geometric (PyG)**| `torch-geometric`| `n/a` | High-performance PyTorch library for deep learning over graph and relational data. |
| **DGL** | `dgl` | `n/a` | Framework-agnostic Graph Neural Network library optimized for large-scale graphs. |

---

# Part III: Specialized Analytical & Modeling Domains

## 11. Time Series Analysis, Forecasting & Anomaly Detection
Statistical forecasting, foundation models, anomaly detection, and temporal ML.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **StatsForecast** | `statsforecast` | `pandas` | Ultra-fast statistical forecasting. C-accelerated AutoARIMA, ETS, Theta, and baseline models. |
| **pmdarima** | `pmdarima` | `pandas` | Auto-ARIMA for Python. Automated statistical grid search for ARIMA/SARIMAX orders with AIC/BIC scoring. |
| **sktime** | `sktime` | `pandas` | Scikit-learn style interfaces for time series classification, regression, and transformation. |
| **aeon** | `aeon` | `pandas` | Dedicated toolkit for time series machine learning, classification, and segmentation. |
| **Prophet** | `prophet` | `pandas` | Additive regression model for time series with strong seasonalities and holiday effects. |
| **Chronos** | `chronos-forecasting` | `both` | Pretrained time series foundation model family for zero-shot forecasting. |
| **TimesFM** | `timesfm` | `both` | Google Research pretrained foundation model for time series forecasting. |
| **NeuralForecast** | `neuralforecast` | `pandas` | Deep learning for time series (N-BEATS, NHITS, PatchTST architectures). |
| **PyTorch Forecasting** | `pytorch-forecasting` | `pandas` | Temporal Fusion Transformer (TFT) and probabilistic forecasting in PyTorch. |
| **Darts** | `darts` | `both` | Unified forecasting library covering both classical statistical and deep neural models. |
| **PyOD** | `pyod` | `both` | Comprehensive toolkit with 40+ outlier and anomaly detection algorithms. |
| **Merlion** | `salesforce-merlion` | `pandas` | End-to-end time series intelligence library for anomaly detection and forecasting. |
| **tslearn** | `tslearn` | `both` | Time series clustering and metric learning using Dynamic Time Warping (DTW). |
| **tsfresh** | `tsfresh` | `pandas` | Automated feature extraction from time series for downstream ML classifiers. |

---

## 12. Causal Inference & Uplift Modeling
Estimating cause-and-effect relationships, heterogeneous treatment effects, and uplift scoring.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **DoWhy** | `dowhy` | `pandas` | End-to-end 4-step causal analysis: model graph, identify estimand, estimate effect, and test refutations. |
| **EconML** | `econml` | `pandas` | Heterogeneous treatment effects and Double/Debiased Machine Learning with causal forests. |
| **CausalML** | `causalml` | `pandas` | Uplift modeling for campaign personalization and A/B test targeting using meta-learners. |
| **CausalNex** | `causalnex` | `pandas` | Bayesian network causal modeling and structure learning via continuous optimization (NOTEARS). |

---

## 13. Survival Analysis & Event History Modeling
Time-to-event estimation, censored data modeling, and hazard rates.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **lifelines** | `lifelines` | `pandas` | Classical survival analysis: Kaplan-Meier, Nelson-Aalen, and Cox Proportional Hazards. |
| **scikit-survival** | `scikit-survival` | `pandas` | Scikit-learn compatible survival models: Random Survival Forests and penalized Cox models. |
| **pycox** | `pycox` | `n/a` | Deep learning survival modeling in PyTorch (DeepSurv, Cox-Time). |

---

## 14. Recommender Systems & Collaborative Filtering
Collaborative filtering, matrix factorization, and deep sequential recommendation.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **implicit** | `implicit` | `both` | Fast GPU/CPU Alternating Least Squares (ALS) and BPR for implicit-feedback interaction datasets. |
| **LightFM** | `lightfm` | `pandas` | Hybrid collaborative and content-based recommendation incorporating user/item metadata. |
| **Surprise** | `scikit-surprise` | `pandas` | Explicit-rating recommendation algorithms (SVD, SVD++, KNN, NMF). |
| **RecBole** | `recbole` | `both` | Unified PyTorch recommendation framework with 100+ benchmark algorithms. |

---

## 15. Geospatial Analytics & Mapping
Vector geometries, satellite rasters, web maps, and street network graphs.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **GeoPandas** | `geopandas` | `pandas` | Spatial operations on tabular data with geometry columns and CRS reprojections. |
| **Shapely** | `shapely` | `n/a` | Planar geometry manipulation for polygons, lines, intersections, and spatial buffers. |
| **Folium** | `folium` | `both` | Leaflet.js interactive web maps rendered directly inside notebook cells. |
| **Rasterio** | `rasterio` | `both` | Satellite and raster imagery processing (GeoTIFFs, windowed reads, spatial band math). |
| **OSMnx** | `osmnx` | `both` | OpenStreetMap road networks, street graph construction, and shortest path routing. |

---

## 16. Bioinformatics, Computational Biology & Genomics
Sequence analysis, structural biology, single-cell genomics, and phylogenetics.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Biopython** | `biopython` | `both` | Sequence I/O (FASTA/GenBank), BLAST parsing, PDB coordinate handling, and sequence motifs. |
| **Scanpy** | `scanpy` | `both` | Single-cell RNA-seq analysis toolkit for clustering, UMAP, and trajectory inference. |
| **AnnData** | `anndata` | `both` | Annotated multi-dimensional matrix container for single-cell and high-throughput omics data. |
| **scikit-bio** | `scikit-bio` | `both` | Biological sequence alignment, phylogenetic trees, and biodiversity metrics. |
| **Biotite** | `biotite` | `both` | 3D structural biology, macromolecular analysis, and PDB/mmCIF processing. |
| **Dna Features Viewer** | `dna-features-viewer`| `both` | Publication-ready visual maps of DNA constructs, plasmids, and genomic annotations. |

---

## 17. Simulation, Agent-Based Modeling & Physics Engines `[Out-of-Pipeline]`
Rigid-body mechanics, emergent behavior, agent simulation, and neural differential equations.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Pymunk** | `pymunk` | `n/a` | 2D rigid-body physics library for collisions, joints, friction, and custom 2D simulation. |
| **Mesa** | `mesa` | `both` | Agent-based modeling (ABM) for simulating emergent behaviors across spatial grids. |
| **PyBullet** | `pybullet` | `n/a` | 3D rigid-body physics engine for robotics, collision detection, and VR simulation. |
| **Brax** | `brax` | `n/a` | Differentiable physics engine developed by Google on JAX for high-throughput robotics simulation. |
| **DeepXDE** | `deepxde` | `n/a` | Physics-Informed Neural Networks (PINNs) solving forward and inverse partial differential equations. |
| **torchdiffeq** | `torchdiffeq` | `n/a` | Differentiable ODE solvers in PyTorch for Neural Ordinary Differential Equations. |

---

# Part IV: Deep Learning, Reinforcement Learning & Foundation Models

## 18. Deep Learning Frameworks, Acceleration & LLM Fine-Tuning
Tensors, distributed training, parameter-efficient fine-tuning (PEFT), and quantization.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **PyTorch** | `torch` | `n/a` | Dynamic tensor computation and neural network framework; pre-installed in Colab. |
| **PyTorch Lightning** | `lightning` | `n/a` | Structured PyTorch engineering decoupling science from hardware engineering. |
| **TensorFlow** | `tensorflow` | `n/a` | End-to-end deep learning platform with Keras API; pre-installed in Colab. |
| **JAX** | `jax[cuda12]` | `n/a` | Composable autodiff transformations and XLA compilation for accelerator computing. |
| **Hugging Face Accelerate** | `accelerate` | `n/a` | Multi-GPU, mixed-precision, and FSDP distributed training wrapper for PyTorch. |
| **PEFT** | `peft` | `n/a` | Parameter-Efficient Fine-Tuning: LoRA, QLoRA, and Prefix Tuning for foundation models. |
| **TRL** | `trl` | `n/a` | Transformer Reinforcement Learning: SFT, DPO, and PPO training loops for LLMs. |
| **bitsandbytes** | `bitsandbytes` | `n/a` | 8-bit and 4-bit quantization primitives enabling QLoRA on Colab GPUs. |
| **Unsloth** | `unsloth` | `n/a` | Memory-efficient LLM fine-tuning library (2-5x faster, 80% less VRAM) tailored for Colab GPUs. |
| **DeepSpeed** | `deepspeed` | `n/a` | Distributed training optimization engine for training very large deep learning models. |
| **torchmetrics** | `torchmetrics` | `n/a` | Standardized PyTorch evaluation metrics supporting multi-GPU distributed synchronization. |
| **timm** | `timm` | `n/a` | Pretrained computer vision neural network backbones and vision transformer architectures. |

---

## 19. Reinforcement Learning (RL) & Simulation
Environments, policy gradient methods, Q-learning, and multi-agent RL.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Gymnasium** | `gymnasium` | `n/a` | Standard API for reinforcement learning environments (official Farama / Gym successor). |
| **Gymnasium Atari** | `gymnasium[atari]` | `n/a` | Standard Atari 2600 benchmark environments via ALE. |
| **Gymnasium Box2D** | `gymnasium[box2d]` | `n/a` | 2D continuous control benchmark environments (LunarLander, BipedalWalker). |
| **Gymnasium MuJoCo** | `gymnasium[mujoco]` | `n/a` | Continuous control robotics benchmark environments. |
| **Stable-Baselines3** | `stable-baselines3` | `n/a` | PyTorch RL algorithm suite (PPO, A2C, DQN, SAC, TD3). |
| **Tianshou** | `tianshou` | `n/a` | Modular PyTorch RL library with fast vectorized replay buffers and modern algorithm support. |
| **Ray RLlib** | `ray[rllib]` | `n/a` | Distributed reinforcement learning scaling across multi-node CPU/GPU clusters. |
| **CleanRL** | `cleanrl` | `n/a` | High-readability, single-file PyTorch implementations of modern RL algorithms. |
| **PettingZoo** | `pettingzoo` | `n/a` | Multi-Agent Reinforcement Learning (MARL) environment standard. |
| **TorchRL** | `torchrl` | `n/a` | Modular PyTorch domain library containing RL building blocks, buffers, and vectorized envs. |

---

## 20. Natural Language Processing, LLMs & Retrieval-Augmented Generation (RAG)
Classic NLP, modern embeddings, vector databases, document parsers, structured outputs, and evaluation.

### Core NLP & Transformers
| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Transformers** | `transformers` | `both` | Pretrained transformer architectures and foundation model hub access. |
| **Datasets** | `datasets` | `both` | Fast Arrow-backed dataset loading and streaming tokenization pipelines. |
| **spaCy** | `spacy` | `both` | Industrial-strength NLP preprocessing: tokenization, NER, POS tagging, and parsing. |
| **NLTK** | `nltk` | `both` | Classic NLP toolkit for tokenization, stemming, lemmatization, and linguistic corpora. |
| **Gensim** | `gensim` | `both` | Topic modeling and document similarity using LDA and Word2Vec. |
| **Stanza** | `stanza` | `both` | Stanford's neural NLP toolkit supporting 60+ human languages. |
| **Flair** | `flair` | `both` | PyTorch NLP framework for state-of-the-art named entity recognition and part-of-speech tagging. |
| **BERTopic** | `bertopic` | `both` | Neural topic modeling leveraging sentence embeddings and class-based TF-IDF. |
| **sentence-transformers** | `sentence-transformers`| `both` | Pretrained dense embedding models for semantic search and clustering. |

### Vector Stores & Information Retrieval
| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **FAISS** | `faiss-cpu` / `faiss-gpu`| `both` | GPU/CPU vector similarity indexing developed by Meta for high-scale nearest neighbors. |
| **ChromaDB** | `chromadb` | `both` | Lightweight embedded vector database ideal for rapid Colab RAG prototyping. |
| **Qdrant** | `qdrant-client` | `both` | Production vector database client with rich payload filtering and hybrid search. |
| **Weaviate** | `weaviate-client` | `both` | Scalable vector database client with native vectorizers and hybrid search. |
| **LanceDB** | `lancedb` | `both` | Serverless vector database built on the Lance columnar data format with zero-copy Arrow reads. |
| **pgvector** | `pgvector` | `both` | Python support for PostgreSQL vector similarity search. |
| **rank_bm25** | `rank-bm25` | `both` | BM25 Okapi lexical search for building hybrid retrieval pipelines (dense + sparse). |
| **Whoosh** | `whoosh` | `both` | Pure-Python full-text indexing and search engine library. |

### Document Parsing for RAG
| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Docling** | `docling` | `n/a` | Deep document parsing converting PDFs, DOCX, and PPTX into clean Markdown with layout preservation. |
| **PyMuPDF** | `pymupdf` | `n/a` | High-speed C-backed PDF text, table, and image extraction. |
| **pdfplumber** | `pdfplumber` | `pandas` | Detailed PDF table and layout extraction with bounding box coordinates. |
| **Unstructured** | `unstructured` | `n/a` | Ingestion preprocessing converting unstructured files (PDF, HTML, emails) into clean document chunks. |

### Structured Outputs, Prompt Optimization & Local Serving
| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Instructor** | `instructor` | `n/a` | Structured LLM extraction coercing responses into strict Pydantic models with automated retries. |
| **Outlines** | `outlines` | `n/a` | Constrained generation enforcing JSON schemas and regex rules via logit masking. |
| **Guidance** | `guidance` | `n/a` | Interleaved generation and control flow framework guaranteeing structured model outputs. |
| **DSPy** | `dspy` | `n/a` | Declarative framework replacing prompt engineering with programmatic teleprompter optimizers. |
| **vLLM** | `vllm` | `n/a` | High-throughput GPU inference engine using PagedAttention for fast batch generation. |
| **llama-cpp-python** | `llama-cpp-python` | `n/a` | Python bindings for llama.cpp supporting GGUF quantized models on CPU and GPU. |
| **Haystack** | `haystack-ai` | `n/a` | Modular end-to-end framework for semantic search and RAG pipelines. |

### RAG Evaluation & Observability
| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **RAGAS** | `ragas` | `pandas` | Evaluation framework quantifying context precision, recall, faithfulness, and answer relevance. |
| **DeepEval** | `deepeval` | `n/a` | Unit testing and evaluation framework for LLM applications and RAG systems. |
| **Langfuse** | `langfuse` | `n/a` | Open-source LLM observability, prompt versioning, and tracing platform. |
| **LangSmith** | `langsmith` | `n/a` | Tracing, debugging, and evaluation platform for LLM chains and agents. |
| **Arize Phoenix** | `arize-phoenix` | `both` | AI observability platform for tracing, evaluations, and embedding visualization. |

---

## 21. Foundation Model SDKs & Multi-Provider Clients
Official model provider SDKs and unified multi-provider routing interfaces.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **openai** | `openai` | `n/a` | Official OpenAI client for completions, embeddings, realtime audio, and structured outputs. |
| **anthropic** | `anthropic` | `n/a` | Official Anthropic client for Claude completions, tool use, and computer use capabilities. |
| **google-genai** | `google-genai` | `n/a` | Official Google Gemini client for multimodal generation, tools, and groundings. |
| **LiteLLM** | `litellm` | `n/a` | Unified OpenAI-compatible interface to call 100+ LLM APIs with standardized routing. |
| **tiktoken** | `tiktoken` | `n/a` | Fast BPE tokenizer library for exact token counting across OpenAI models. |

---

## 22. AI Agents & Autonomous Multi-Agent Frameworks
Frameworks for building tool-using assistants, cyclic graphs, and collaborative multi-agent teams.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **LangGraph** | `langgraph` | `n/a` | Stateful, cyclic multi-agent graph orchestration supporting human-in-the-loop and branching. |
| **Pydantic AI** | `pydantic-ai` | `n/a` | Type-safe production agent framework built by Pydantic with dependency injection. |
| **smolagents** | `smolagents` | `n/a` | Lightweight Hugging Face framework where agents write and execute executable Python code actions. |
| **OpenAI Agents SDK** | `openai-agents` | `n/a` | Production multi-agent orchestration for building agent handoffs and guardrails. |
| **Model Context Protocol (mcp)** | `mcp` | `n/a` | Standardized Python SDK for building tool servers and clients over the open MCP protocol. |
| **LangChain** | `langchain` | `n/a` | Framework for prompt templating, memory, document loaders, and composable LLM chains. |
| **LlamaIndex** | `llama-index` | `n/a` | Data framework for building RAG applications and context-connected knowledge agents. |
| **CrewAI** | `crewai` | `n/a` | Role-based autonomous agent framework organizing specialized teams of collaborating agents. |
| **AutoGen** | `autogen-agentchat` | `n/a` | Multi-agent conversation platform enabling collaborative code execution and problem solving. |
| **Semantic Kernel** | `semantic-kernel` | `n/a` | Enterprise AI orchestrator integrating models with conventional programming languages. |

---

## 23. Computer Vision, Spatial AI & Generative Media
Object detection, segmentation, optical character recognition (OCR), pose tracking, and diffusion models.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Ultralytics (YOLO)** | `ultralytics` | `both` | Real-time object detection, instance segmentation, bounding box classification, and tracking. |
| **Supervision** | `supervision` | `both` | Computer vision utilities for polygon annotators, detections filtering, and box smoothing. |
| **Segment Anything (SAM)** | `segment-anything` | `n/a` | Foundation segmentation model for zero-shot promptable visual mask extraction. |
| **OpenCLIP** | `open-clip-torch` | `n/a` | Vision-language representations for zero-shot image classification and semantic image search. |
| **Diffusers** | `diffusers` | `n/a` | Generative diffusion model pipelines for latent image generation and image-to-image synthesis. |
| **Kornia** | `kornia` | `n/a` | Differentiable computer vision algorithms built natively on top of PyTorch. |
| **Detectron2** | `detectron2` | `n/a` | Advanced object detection and instance segmentation research platform by Meta. |
| **MediaPipe** | `mediapipe` | `n/a` | Real-time on-device facial mesh, iris tracking, hand landmarking, and pose tracking. |
| **EasyOCR** | `easyocr` | `both` | Ready-to-use Optical Character Recognition (OCR) supporting 80+ languages. |
| **pytesseract** | `pytesseract` | `both` | Python wrapper for Google's Tesseract-OCR engine. |
| **PaddleOCR** | `paddleocr` | `both` | Practical ultra-lightweight OCR system supporting multilingual text detection and recognition. |
| **OpenCV** | `opencv-python` | `both` | Real-time computer vision operations: reading/writing, color transforms, contours, and video. |
| **Albumentations** | `albumentations` | `both` | Fast image and bounding box augmentations optimized for deep learning pipelines. |
| **Pillow (PIL)** | `pillow` | `both` | Core Python imaging library for basic resizing, cropping, and format conversions; pre-installed. |

---

## 24. Audio, Speech & Video Processing
Speech recognition, speaker diarization, audio spectrogram analysis, and programmatic video editing.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **openai-whisper** | `openai-whisper` | `both` | General-purpose speech recognition model for transcription and language translation. |
| **faster-whisper** | `faster-whisper` | `both` | Reimplementation of Whisper using CTranslate2 for up to 4x faster GPU inference. |
| **pyannote.audio** | `pyannote.audio` | `both` | Speaker diarization and voice activity detection in multi-speaker audio recordings. |
| **librosa** | `librosa` | `both` | Audio signal analysis extracting Mel spectrograms, MFCCs, chroma features, and tempo. |
| **torchaudio** | `torchaudio` | `n/a` | GPU-accelerated audio processing and acoustic modeling in PyTorch; pre-installed in Colab. |
| **pydub** | `pydub` | `n/a` | High-level audio slicing, concatenation, and export across MP3, WAV, and OGG formats. |
| **Piper TTS** | `piper-tts` | `n/a` | Fast, lightweight local neural text-to-speech synthesis engine. |
| **MoviePy** | `moviepy` | `both` | Programmatic video editing: cutting, concatenating, title overlays, and audio track extraction. |
| **ffmpeg-python** | `ffmpeg-python` | `both` | Complex FFmpeg audio/video filter graph creation and streaming bindings. |
| **PyAV** | `av` | `both` | Direct, fast Pythonic bindings to FFmpeg's C libraries for decoding video frames into arrays. |

---

# Part V: MLOps, Governance, Privacy & Serving

## 25. Model Interpretability, Fairness & Data Drift (XAI & Auditing)
Feature attributions, glass-box models, counterfactuals, algorithmic fairness, and production drift monitoring.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **SHAP** | `shap` | `both` | Game-theoretic feature attribution (TreeSHAP, KernelSHAP); global beeswarm and waterfall plots. |
| **LIME** | `lime` | `both` | Local surrogate explanations approximating black-box model decisions for individual samples. |
| **InterpretML** | `interpret` | `pandas` | Explainable Boosting Machines (EBM) providing exact glass-box interpretability with GBDT performance. |
| **Alibi** | `alibi` | `both` | Machine learning model inspection: counterfactual explanations, prototypes, and anchor rules. |
| **Fairlearn** | `fairlearn` | `pandas` | Algorithmic fairness assessment quantifying demographic parity and equalized odds with mitigations. |
| **Evidently** | `evidently` | `pandas` | Interactive reports and metrics for tracking data drift, target drift, and model performance. |
| **NannyML** | `nannyml` | `pandas` | Model performance estimation in production without requiring immediate target labels. |
| **TransformerLens** | `transformer-lens` | `n/a` | Mechanistic interpretability for reverse-engineering attention heads and residual streams in LLMs. |
| **Captum** | `captum` | `n/a` | PyTorch model interpretability library implementing integrated gradients and DeepLIFT. |

---

## 26. Hyperparameter Optimization & Experiment Tracking
Automated tuning engines, artifact registries, and configuration management.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Optuna** | `optuna` | `both` | Next-generation hyperparameter optimization using TPE, CMA-ES, and automated pruning. |
| **Ray Tune** | `ray[tune]` | `both` | Distributed hyperparameter optimization supporting parallel trials and early stopping algorithms. |
| **Ax** | `ax-platform` | `both` | Adaptive experimentation platform for Bayesian optimization powered by PyTorch (BoTorch). |
| **Weights & Biases** | `wandb` | `both` | Cloud experiment tracking, artifact versioning, hyperparameter sweeps, and live metric curves. |
| **MLflow** | `mlflow` | `both` | Open-source ML lifecycle tracking covering experiments, model registries, and artifacts. |
| **ClearML** | `clearml` | `both` | Full-stack MLOps suite for experiment tracking, pipeline orchestration, and remote execution. |
| **TensorBoard** | `tensorboard` | `n/a` | Interactive visualization dashboard for loss curves and histograms; pre-installed (`%load_ext tensorboard`). |
| **DVC** | `dvc` | `n/a` | Data Version Control: Git-like versioning for multi-gigabyte datasets and models. |
| **Hydra** | `hydra-core` | `n/a` | Hierarchical experiment configuration framework managing parameters via composable YAML files. |

---

## 27. Federated Learning & Privacy-Preserving ML `[Out-of-Pipeline]`
Decentralized collaborative training, differential privacy (DP), and secure computation.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **Flower** | `flwr` | `n/a` | Framework-agnostic federated learning with built-in virtual client simulation for single Colab notebooks. |
| **Opacus** | `opacus` | `n/a` | Differential privacy for PyTorch implementing DP-SGD with per-sample clipping and privacy accounting. |
| **PySyft** | `syft` | `n/a` | OpenMined platform for secure, privacy-preserving remote data science. |
| **diffprivlib** | `diffprivlib` | `pandas` | IBM's library offering differential privacy wrappers for standard scikit-learn models. |
| **Microsoft Presidio** | `presidio-analyzer` | `n/a` | Sensitive data identification and PII anonymization for text and structured records. |

---

## 28. Model Serving, Interactive UIs & Inference Engines `[Out-of-Pipeline]`
High-throughput inference runtimes, REST APIs, and interactive demonstration interfaces.

| Library | pip package | Engine Compatibility | Primary Use Case & Notes |
|---|---|---|---|
| **ONNX Runtime** | `onnxruntime` | `n/a` | Hardware-accelerated inference engine optimizing models across CPUs and NVIDIA GPUs. |
| **Streamlit** | `streamlit` | `both` | Turns Python scripts into shareable interactive web applications with minimal frontend code. |
| **Gradio** | `gradio` | `both` | Creates intuitive web UIs for machine learning models; widely used for Hugging Face Spaces demos. |
| **FastAPI** | `fastapi[standard]` | `n/a` | Asynchronous high-performance web API framework for serving model inference with OpenAPI docs. |
