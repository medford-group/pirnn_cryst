"""Training loop and evaluation utilities for the PIRNN sweep."""

import torch
import numpy as np

from .constants import GT_PARAMS
from .models import PBM, PIRNN
from .losses import lossX, lossG, lossSmooth
from .utils import set_seed


def calculate_param_error(rhs, device):
    """RMSRE between learned (exp of log-params) and ground-truth kinetics."""
    params_pred = torch.exp(torch.tensor([
        rhs.kb2.item(), rhs.alfa.item(), rhs.beta.item(),
        rhs.kg.item(), rhs.Eag.item(), rhs.gama_g.item(),
    ], device=device))

    params_true = torch.tensor([
        GT_PARAMS['kb2'], GT_PARAMS['alfa'], GT_PARAMS['beta'],
        GT_PARAMS['kg'], GT_PARAMS['Eag'], GT_PARAMS['gama_g'],
    ], device=device)

    relative_errors = (params_pred - params_true) / params_true
    return torch.sqrt(torch.mean(relative_errors ** 2)).item()


def evaluate_model(model, rhs, x0, T_data, y_data,
                   eval_physics_in_val=True):
    """Evaluate model on a dataset, returning loss components.

    Parameters
    ----------
    eval_physics_in_val : bool
        If True, validation loss = lX + physics_lambda * lG (noise sweep style).
        If False, validation loss = lX only (solubility-shift style).
    """
    model.eval()
    with torch.no_grad():
        pred = model(x0, T_data)
        lX = lossX(pred, y_data)
        lG = lossG(pred, T_data, rhs, model.delta)
        lSmooth_val = lossSmooth(pred)
        if eval_physics_in_val:
            loss = lX + lG + lSmooth_val
        else:
            loss = lX
    return loss.item(), lX.item(), lG.item(), lSmooth_val.item()


def train_sweep(data, device, data_size_sweep, physics_lambda=1e1,
                epochs=400000, seed=42, eval_physics_in_val=False,
                log_interval=500, print_interval=2000, dropout=0.2):
    """Run the full data-size sweep training loop.

    Parameters
    ----------
    data : dict
        Output of :func:`generate_synthetic_data`
    device : str
        Torch device
    data_size_sweep : list[int]
        List of training-set sizes to sweep over
    physics_lambda : float
        Weight on the physics loss term during training
    epochs : int
        Number of training epochs per data size
    seed : int
        Random seed
    eval_physics_in_val : bool
        Include physics loss in validation metric
    log_interval : int
        Record loss history every this many epochs
    print_interval : int
        Print progress every this many epochs
    dropout : float
        Dropout rate for PIRNN

    Returns
    -------
    results : list[dict]
        Summary per data size.
    loss_history : list[dict]
        Detailed per-epoch loss records.
    best_val_model : dict
        Best model checkpoints of the validation set.
    """
    y_scale = data['y_scale']

    # Tensors for training
    x0 = torch.tensor(data['initial_conditions'], dtype=torch.float32).to(device)
    T_data = torch.tensor(data['T_input'], dtype=torch.float32).to(device)
    y_data = torch.tensor(data['y_true'], dtype=torch.float32).to(device)

    # Tensors for test
    x0_test = torch.tensor(data['initial_conditions_test'], dtype=torch.float32).to(device)
    T_data_test = torch.tensor(data['T_input_test'], dtype=torch.float32).to(device)
    y_data_test = torch.tensor(data['y_true_test'], dtype=torch.float32).to(device)

    # No-noise versions (may be same as noisy if noise_std == 0)
    y_data_no_noise = torch.tensor(data['y_true_no_noise'], dtype=torch.float32).to(device)
    y_data_test_no_noise = torch.tensor(data['y_true_test_no_noise'], dtype=torch.float32).to(device)

    # Train/val split
    torch.manual_seed(42)
    num_total = x0.shape[0]
    val_size = 20
    shuffled = torch.randperm(num_total)
    train_idx = shuffled[:num_total - val_size]
    val_idx = shuffled[num_total - val_size:]

    x0_train = x0[train_idx]
    T_data_train = T_data[train_idx]
    y_data_train = y_data[train_idx]

    x0_val = x0[val_idx]
    T_data_val = T_data[val_idx]
    y_data_val = y_data[val_idx]
    y_data_val_no_noise = y_data_no_noise[val_idx]

    results = []
    loss_history = []
    best_val_model = {}

    for size in data_size_sweep:
        x0_subset = x0_train[:size]
        T_data_subset = T_data_train[:size]
        y_data_subset = y_data_train[:size]

        print("-" * 50)
        print(f"TRAINING CONFIG: Data Size = {size}")
        print("-" * 50)

        set_seed(seed)

        rhs = PBM(y_scale=y_scale).to(device)
        model = PIRNN(dropout=dropout).to(device)
        opt = torch.optim.Adam(
            list(model.parameters()) + list(rhs.parameters()), lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-7
        )

        best_val_loss = float('inf')
        best_val_index = 0

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()

            pred = model(x0_subset, T_data_subset)
            lX = lossX(pred, y_data_subset)
            lG = lossG(pred, T_data_subset, rhs, model.delta)
            lSmooth = lossSmooth(pred)
            loss = lX + physics_lambda * lG + lSmooth

            loss.backward()
            opt.step()
            scheduler.step()

            if epoch % log_interval == 0 or epoch == epochs - 1:
                val_loss, val_lX, val_lG, val_lSmooth = evaluate_model(
                    model, rhs, x0_val, T_data_val, y_data_val,
                    eval_physics_in_val,
                )
                test_loss, test_lX, test_lG, test_lSmooth = evaluate_model(
                    model, rhs, x0_test, T_data_test, y_data_test,
                    eval_physics_in_val,
                )

                # No-noise evaluation
                val_loss_nn, val_lX_nn, val_lG_nn, val_lSmooth_nn = evaluate_model(
                    model, rhs, x0_val, T_data_val, y_data_val_no_noise,
                    eval_physics_in_val,
                )
                test_loss_nn, test_lX_nn, test_lG_nn, test_lSmooth_nn = evaluate_model(
                    model, rhs, x0_test, T_data_test, y_data_test_no_noise,
                    eval_physics_in_val,
                )

                param_error = calculate_param_error(rhs, device)

                loss_history.append({
                    'data_size': size,
                    'epoch': epoch,
                    'train_loss': loss.item(),
                    'train_lX': lX.item(),
                    'train_lG': lG.item(),
                    'train_lSmooth': lSmooth.item(),
                    'val_loss': val_loss, 'val_lX': val_lX,
                    'val_lG': val_lG, 'val_lSmooth': val_lSmooth,
                    'val_loss_no_noise': val_loss_nn, 'val_lX_no_noise': val_lX_nn,
                    'val_lG_no_noise': val_lG_nn, 'val_lSmooth_no_noise': val_lSmooth_nn,
                    'test_loss': test_loss, 'test_lX': test_lX,
                    'test_lG': test_lG, 'test_lSmooth': test_lSmooth,
                    'test_loss_no_noise': test_loss_nn, 'test_lX_no_noise': test_lX_nn,
                    'test_lG_no_noise': test_lG_nn, 'test_lSmooth_no_noise': test_lSmooth_nn,
                    'rhs_kb2': rhs.kb2.item(),
                    'rhs_alfa': rhs.alfa.item(),
                    'rhs_beta': rhs.beta.item(),
                    'rhs_kg': rhs.kg.item(),
                    'rhs_Eag': rhs.Eag.item(),
                    'rhs_gama_g': rhs.gama_g.item(),
                    'param_error_rmsre': param_error,
                    'noise_scale': model.noise_scale.item(),
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_index = epoch
                    best_val_model[str(size)] = {
                        'model_state': model.state_dict(),
                        'rhs_state': rhs.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                    }

                if epoch % print_interval == 0 or epoch == epochs - 1:
                    print(
                        f'Epoch {epoch:6d}: '
                        f'Train Loss={loss.item():.4e} '
                        f'(lX={lX.item():.4e}, lG={lG.item():.4e}) | '
                        f'Val Loss={val_loss:.4e} | '
                        f'Test Loss={test_loss:.4e} | '
                        f'LR={opt.param_groups[0]["lr"]:.2e}'
                    )

        last = loss_history[-1]
        print("\nTraining finished.")
        print(f"Final Val Loss: {last['val_loss']:.4e} "
              f"(lX={last['val_lX']:.4e}, lG={last['val_lG']:.4e})")
        print(f"Final Test Loss: {last['test_loss']:.4e} "
              f"(lX={last['test_lX']:.4e}, lG={last['test_lG']:.4e})")
        print(f"Final Parameter Error (RMSRE): {last['param_error_rmsre']:.4e}")

        results.append({
            'data_size': size,
            'epochs': epochs,
            'final_val_loss': last['val_loss'],
            'final_test_loss': last['test_loss'],
            'final_val_loss_no_noise': last['val_loss_no_noise'],
            'final_test_loss_no_noise': last['test_loss_no_noise'],
            'best_val_loss': best_val_loss,
            'best_val_index': best_val_index,
        })

    return results, loss_history, best_val_model
