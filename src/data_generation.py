"""Synthetic crystallization data generation.

Generates temperature profiles and solves the population balance ODEs to 
produce synthetic training/test data with configurable noise and
solubility shifts.
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .constants import kv, rho, R, GT_PARAMS
from .utils import set_seed


# ---------------------------------------------------------------------------
# Temperature profile generation
# ---------------------------------------------------------------------------

def make_profile(Tplat, final_T=0.0, plateau_dur=120, cooling_rate=-0.25, hold_time=500):
    """Create a three-stage cooling crystallization temperature profile.

    Stages:
        1. Constant-temperature plateau at *Tplat* for *plateau_dur* minutes.
        2. Linear cooling at *cooling_rate* (deg C / min) to *final_T*.
        3. Hold at *final_T* until *hold_time*.

    Returns a DataFrame with columns ``t_min`` and ``T_C``.
    """
    # Stage 1: plateau
    t1 = np.arange(0, plateau_dur + 1)
    T1 = np.full_like(t1, Tplat, dtype=float)

    # Stage 2: linear cooling
    dT2 = np.abs(final_T - Tplat)
    t2_len = dT2 / abs(cooling_rate)
    t2_start = t1[-1] + 1
    t2 = np.arange(t2_start, t2_start + t2_len + 1)
    T2 = Tplat + cooling_rate * (t2 - t2[0])

    t = np.concatenate([t1, t2])
    T = np.concatenate([T1, T2])

    # Stage 3: hold if the profile hasn't already reached hold_time
    if t[-1] < hold_time:
        t3_start = t[-1] + 1
        t3 = np.arange(t3_start, hold_time + 1)
        T3 = np.full_like(t3, final_T, dtype=float)
        t = np.concatenate([t, t3])
        T = np.concatenate([T, T3])

    return pd.DataFrame({'t_min': t.astype(int), 'T_C': T})


def generate_runs(n=100, seed=42):
    """Generate *n* randomised cooling profiles and initial concentrations.

    Returns
    -------
    runs : dict[str, DataFrame]
        Mapping ``"run_1"`` ... ``"run_n"`` to temperature-profile DataFrames.
    meta : DataFrame
        Per-run metadata (Tplat, cooling rate, initial concentration, plateau duration).
    """
    rng = np.random.default_rng(seed)

    # Change ranges if want different profile characteristics
    Tplat = rng.uniform(30, 50, n)
    cooling_rate = rng.uniform(-0.6, -0.15, n)
    C0 = rng.uniform(0.37, 0.5, n)
    plateau_dur = rng.uniform(80, 140, n)

    runs = {
        f"run_{i+1}": make_profile(Tplat=Tplat[i], plateau_dur=plateau_dur[i], cooling_rate=cooling_rate[i])
        for i in range(n)
    }

    meta = pd.DataFrame({
        "run_id": [f"run_{i+1}" for i in range(n)],
        "Tplat": Tplat,
        "cooling_rate": cooling_rate,
        "C0": C0,
        "plateau_dur": plateau_dur,
    })

    return runs, meta


# ---------------------------------------------------------------------------
# ODE system (ground-truth)
# ---------------------------------------------------------------------------

def ceq_polynomial(T_K, solubility_shift=1.0):
    """Equilibrium solubility as a polynomial in temperature [K]."""
    return (
        -16.17
        + 1.765e-1 * T_K
        - 6.439e-4 * T_K ** 2
        + 7.915e-7 * T_K ** 3
    ) * solubility_shift


def odefun(t, x, T, solubility_shift=1.0):
    """Population balance equation using ground-truth params."""
    mu0, mu1, mu2, mu3, conc = x

    kb2 = GT_PARAMS['kb2']
    alfa = GT_PARAMS['alfa']
    beta = GT_PARAMS['beta']
    kg = GT_PARAMS['kg']
    Eag = GT_PARAMS['Eag']
    gama_g = GT_PARAMS['gama_g']

    ms = kv * rho * mu3 * 1e3

    T_K = T + 273.15
    Ceq = ceq_polynomial(T_K, solubility_shift)

    S = conc / Ceq

    B2 = kb2 * (max(S - 1, 0) ** alfa) * (ms ** beta)
    G = kg * np.exp(-Eag / (R * T_K)) * max(conc - Ceq, 0) ** gama_g

    dmi0dt = B2
    dmi1dt = G * mu0
    dmi2dt = 2 * G * mu1
    dmi3dt = 3 * G * mu2
    dcdt = -3 * rho * kv * G * mu2

    return [dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt]


# ---------------------------------------------------------------------------
# Full synthetic dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(runs, meta, noise_std=0.0, solubility_shift=1.0,
                            n_train=80, seed=42):
    """Solve the ODE for each run to generate the complete synthetic dataset.

    Parameters
    ----------
    runs : dict
        Temperature-profile DataFrames from :func:`generate_runs`.
    meta : DataFrame
        Metadata used to populate the ``C0`` column.
    noise_std : float
        Noise level as a fraction of each state's standard deviation
        (0.0 = noiseless, 0.1 = 10%, etc.).
    solubility_shift : float
        Multiplicative shift on the equilibrium solubility (1.0 = no shift).
    n_train : int
        Number of runs allocated to training&validation (remainder goes to test).
    seed : int
        Random seed for train/test split and noise generation.

    Returns
    -------
    dict with keys:
        t_input, T_input, y_true, y_true_no_noise, initial_conditions,
        t_input_test, T_input_test, y_true_test, y_true_test_no_noise,
        initial_conditions_test, y_scale, train_exp, test_exp
    """
    set_seed(seed)

    n = len(runs)
    C0 = meta['C0'].values

    selected = np.random.choice(np.arange(1, n + 1), size=n_train, replace=False)
    all_nums = np.arange(1, n + 1)
    remaining = np.setdiff1d(all_nums, selected)

    train_exp = list(selected)
    test_exp = list(remaining)

    t_input, T_input, y_true_list, y_true_no_noise_list, ic_list = [], [], [], [], []
    t_input_test, T_input_test, y_true_test_list, y_true_test_no_noise_list, ic_test_list = [], [], [], [], []

    for i in range(1, n + 1):
        df_exp = runs[f'run_{i}']
        time_eval = df_exp['t_min'].to_numpy().astype(float)
        time_span = (time_eval[0], time_eval[-1])
        temperature = df_exp['T_C'].to_numpy()
        dx0 = np.array([100., 0., 0., 0., C0[i - 1]])

        # Interpolate temperature for ODE solver
        T_interp = interp1d(time_eval, temperature, kind='linear', fill_value="extrapolate")

        def _odefun_interp(t, x, _T_interp=T_interp):
            return odefun(t, x, _T_interp(t), solubility_shift)

        sol = solve_ivp(_odefun_interp, time_span, dx0, t_eval=time_eval, method="LSODA")

        time_sol = sol.t - np.min(sol.t)

        # Add noise
        if noise_std > 0.0:
            mu = sol.y[:4, :]
            conc_sol = sol.y[4, :]

            conc_std_val = np.std(conc_sol)
            mu_stds = np.std(mu, axis=1, keepdims=True)

            conc_noise = np.random.normal(0.0, noise_std * conc_std_val, size=conc_sol[1:].shape)
            mu_noise = np.random.normal(0.0, noise_std * mu_stds, size=mu[:, 1:].shape)

            conc_noisy = conc_sol[1:] + conc_noise
            mu_noisy = mu[:, 1:] + mu_noise

            sol_noise = np.vstack((mu_noisy, conc_noisy))
            sol_noise = np.hstack((dx0[:, None], sol_noise)) # add back initial condition (no noise on IC)
        else:
            sol_noise = sol.y # no noise added

        # Append to train/test lists
        if i in train_exp:
            t_input.append(time_sol.reshape(-1, 1))
            T_input.append(temperature.reshape(-1, 1))
            ic_list.append(dx0)
            y_true_list.append(sol_noise.T)
            y_true_no_noise_list.append(sol.y.T)
        elif i in test_exp:
            t_input_test.append(time_sol.reshape(-1, 1))
            T_input_test.append(temperature.reshape(-1, 1))
            ic_test_list.append(dx0)
            y_true_test_list.append(sol_noise.T)
            y_true_test_no_noise_list.append(sol.y.T)


    y_true = np.vstack(y_true_list)
    y_scale = np.max(y_true, axis=0) # scale comes from training data only
    y_true = y_true / y_scale
    y_true = y_true.reshape(len(train_exp), -1, y_true.shape[1])

    y_true_no_noise = np.vstack(y_true_no_noise_list)
    y_true_no_noise = y_true_no_noise / y_scale
    y_true_no_noise = y_true_no_noise.reshape(len(train_exp), -1, y_true_no_noise.shape[1])

    initial_conditions = np.vstack(ic_list) / y_scale

    T_input_arr = np.vstack(T_input)
    T_input_arr = T_input_arr.reshape(len(train_exp), -1, 1)

    t_input_arr = np.vstack(t_input)

    # Test set
    y_true_test = np.vstack(y_true_test_list)
    y_true_test = y_true_test / y_scale
    y_true_test = y_true_test.reshape(len(test_exp), -1, y_true_test.shape[1])

    y_true_test_no_noise = np.vstack(y_true_test_no_noise_list)
    y_true_test_no_noise = y_true_test_no_noise / y_scale
    y_true_test_no_noise = y_true_test_no_noise.reshape(len(test_exp), -1, y_true_test_no_noise.shape[1])

    initial_conditions_test = np.vstack(ic_test_list) / y_scale

    T_input_test_arr = np.vstack(T_input_test)
    T_input_test_arr = T_input_test_arr.reshape(len(test_exp), -1, 1)

    t_input_test_arr = np.vstack(t_input_test)

    return {
        't_input': t_input_arr,
        'T_input': T_input_arr,
        'y_true': y_true,
        'y_true_no_noise': y_true_no_noise,
        'initial_conditions': initial_conditions,
        't_input_test': t_input_test_arr,
        'T_input_test': T_input_test_arr,
        'y_true_test': y_true_test,
        'y_true_test_no_noise': y_true_test_no_noise,
        'initial_conditions_test': initial_conditions_test,
        'y_scale': y_scale,
        'train_exp': train_exp,
        'test_exp': test_exp,
    }
