"""Post-training evaluation: re-integrate ODE with learned parameters."""

import warnings
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm

from .constants import kv, rho, R


def odefun_pred(t, x, T_func, params):
    """ODE RHS using *learned* (exponentiated) parameters for re-evaluation."""
    mu0, mu1, mu2, mu3, conc = x
    kb2, alfa, beta, kg, Eag, gama_g = np.exp(params)

    ms = kv * rho * mu3 * 1e3

    T = T_func(t)
    T_K = T + 273.15
    Ceq = (
        -16.17
        + 1.765e-1 * T_K
        - 6.439e-4 * T_K ** 2
        + 7.915e-7 * T_K ** 3
    )

    S = conc / Ceq

    B2 = kb2 * (max(S - 1, 0) ** alfa) * (ms ** beta)
    G = kg * np.exp(-Eag / (R * T_K)) * max(conc - Ceq, 0) ** gama_g

    dmi0dt = B2
    dmi1dt = G * mu0
    dmi2dt = 2 * G * mu1
    dmi3dt = 3 * G * mu2
    dcdt = -3 * rho * kv * G * mu2

    return [dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt]


def reevaluate_params_on_test(loss_history_df, data, eval_epoch_step=5000,
                              max_epoch=400000):
    """Re-integrate the ODE with learned params at selected epochs and
    compute MSE on the test set.

    Parameters
    ----------
    loss_history_df : DataFrame
        Must have columns ``epoch``, ``data_size``, and ``rhs_*`` param columns.
    data : dict
        Output of :func:`generate_synthetic_data`.

    Returns
    -------
    re_eval_results : list[dict]
        Each entry has ``data_size``, ``epoch``, and ``re_eval_test_mse``.
    """
    rhs_param_names = ['rhs_kb2', 'rhs_alfa', 'rhs_beta', 'rhs_kg', 'rhs_Eag', 'rhs_gama_g']

    target_epochs = list(np.arange(eval_epoch_step, max_epoch, eval_epoch_step, dtype=int))
    df_to_evaluate = loss_history_df[loss_history_df['epoch'].isin(target_epochs)].copy()
    print(f"Re-evaluating {len(df_to_evaluate)} parameter sets "
          f"across {len(target_epochs)} target epochs.")

    t_input_test = data['t_input_test']
    T_input_test = data['T_input_test']
    y_true_test = data['y_true_test']
    initial_conditions_test = data['initial_conditions_test']
    y_scale = data['y_scale']
    test_exp = data['test_exp']

    n_time = t_input_test.shape[0] // len(test_exp) if t_input_test.ndim == 2 else t_input_test.shape[1]

    re_eval_results = []

    for _, row in tqdm(df_to_evaluate.iterrows(), total=df_to_evaluate.shape[0]):
        params = row[rhs_param_names].values.astype(float)
        total_mse = 0

        for i in range(len(test_exp)):
            time_eval = np.arange(n_time, dtype=float)
            time_span = (time_eval[0], time_eval[-1])
            y0 = initial_conditions_test[i] * y_scale
            T_profile = T_input_test[i].flatten()
            y_true_profile = (y_true_test[i] * y_scale).T

            T_func = interp1d(time_eval, T_profile, kind='linear',
                              fill_value="extrapolate")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sol = solve_ivp(
                    fun=odefun_pred,
                    t_span=time_span,
                    y0=y0,
                    t_eval=time_eval,
                    args=(T_func, params),
                    method='LSODA',
                )

            y_pred = (sol.y.T / y_scale).T
            if y_pred.shape == y_true_profile.shape:
                mse = np.mean((y_pred - (y_true_profile.T / y_scale).T) ** 2)
                total_mse += mse
            else:
                total_mse += np.nan

        re_eval_results.append({
            'data_size': row['data_size'],
            'epoch': row['epoch'],
            're_eval_test_mse': total_mse / len(test_exp),
        })

    return re_eval_results
