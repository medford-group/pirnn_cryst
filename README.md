# PIRNNs for batch crystallization under uncertainties

A physics-informed recurrent neural network (PIRNN) framework for modeling and simulating batch cooling crystallization. This repo is the official implementation of the paper *"Modeling Batch Crystallization under Uncertainty Using Physics-informed Machine Learning"* by D. Nai *et al*. This repository provides importable essential functions and a Jupyter notebook to perform configurable sweeps.

## Info
Authors: Dingqi Nai, Huayu Li, Martha A. Grover, Andrew J. Medford

Email correspondence: [dnai3@gatech.edu](mailto:dnai3@gatech.edu), [ajm@gatech.edu](mailto:ajm@gatech.edu)

Feel free to contact the email correspondence if you face any problems implementing the framework.

## Repository Structure

```
├── src/                        # Importable Python package
│   ├── constants.py            # Constants & ground-truth PBM parameters
│   ├── utils.py                # Random seed setting for reproducibility, device selection
│   ├── data_generation.py      # Temperature profiles and synthetic data generation
│   ├── models.py               # PBM (Learnable physics model), PIRNN (framework core)
│   ├── losses.py               # Loss terms (MSE, Huber, Smoothness, Physics)
│   ├── training.py             # Training loop
│   └── evaluation.py           # ODE re-evaluation with learned parameters
├── notebooks/
│   ├── sweep.ipynb             # Clean configurable sweep notebook
├── figures/                    # Notebooks for figure generation
│   ├── plots_noise.ipynb       # Figures for different noise levels
│   ├── plots_sampling.ipynb    # Figures for different sampling frequencies
│   └── plots_solushift.ipynb   # Figures for model mismatch
├── sweep_results/              # Output directory
│   ├── metrics_log/            # Recorded loss history and ODE re-evaluation during the training loop (.csv)
│   ├── saved_data/             # Example synthetic training and testing data (.npz)
│   └── saved_model/            # Saved best performance model on validation data (.pth)
├── requirements.txt
└── README.md
```

## Dependencies

This project depends on the following packages:

- [PyTorch](https://github.com/pytorch/pytorch) (>=2.10.0)
- [NumPy](https://github.com/numpy/numpy) (>=1.24.3)
- [SciPy](https://scipy.org/) (>=1.10.1)
- [Matplotlib](https://matplotlib.org/) (>=3.7.1)
- [pandas](https://pandas.pydata.org/) (>=2.2.0)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) (>=1.2.2)
- [tqdm](https://github.com/tqdm/tqdm)
