"""Neural network models for physics-informed crystallization RNN."""

import torch
import torch.nn as nn

from .constants import R, rho, kv


class PBM(nn.Module):
    """Learnable right-hand side of the crystallization PBM.

    All six kinetic parameters (kb2, alfa, beta, kg, Eag, gama_g) are stored
    in log-space as ``nn.Parameter`` and exponentiated at evaluation time.
    """

    def __init__(self, y_scale=None):
        super().__init__()

        self.R = R
        self.rho = rho
        self.kv = kv

        self._y_scale = y_scale

        # Learnable kinetic parameters (log-space)
        self.kb2 = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.alfa = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.kg = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.Eag = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.gama_g = nn.Parameter(torch.tensor(1, dtype=torch.float32))

    @property
    def y_scale(self):
        return self._y_scale

    @y_scale.setter
    def y_scale(self, value):
        self._y_scale = value

    def Ceq(self, T_K):
        """Equilibrium solubility polynomial (no shift applied here)."""
        return (
            -16.17
            + 1.765e-1 * T_K
            - 6.439e-4 * T_K ** 2
            + 7.915e-7 * T_K ** 3
        )

    def forward(self, t, x, T):
        relu = torch.nn.functional.relu
        ys = self.y_scale

        mu0 = x[..., 0] * ys[0]
        mu1 = x[..., 1] * ys[1]
        mu2 = x[..., 2] * ys[2]
        mu3 = x[..., 3] * ys[3]
        C = x[..., 4] * ys[4]

        T_K = T + 273.15
        Ceq_val = self.Ceq(T_K)

        S = C / torch.clamp(Ceq_val, min=1e-16)
        ms = self.kv * self.rho * mu3 * 1e3

        B2 = (torch.exp(self.kb2)
              * relu(S - 1.0).pow(torch.exp(self.alfa))
              * ms.pow(torch.exp(self.beta)))

        G = (torch.exp(self.kg)
             * torch.exp(-(torch.exp(self.Eag) / (self.R * T_K)))
             * relu(C - Ceq_val).pow(torch.exp(self.gama_g)))

        dmu0 = B2 / ys[0]
        dmu1 = G * mu0 / ys[1]
        dmu2 = 2 * G * mu1 / ys[2]
        dmu3 = 3 * G * mu2 / ys[3]
        dC = -3 * self.rho * self.kv * G * mu2 / ys[4]

        return torch.stack([dmu0, dmu1, dmu2, dmu3, dC], dim=-1)


class PIRNN(nn.Module):
    """Physics-Informed Recurrent Neural Network for state prediction.

    Uses an LSTM backbone with batch normalisation and a softplus output
    to ensure non-negative state predictions.
    """

    def __init__(self, n_state=5, n_ctrl=1, hidden=64, layers=2, delta=1, dropout=0.2):
        super().__init__()
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.delta = delta
        self.dropout_layer = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=n_state + n_ctrl,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.batch_norm = nn.BatchNorm1d(hidden)
        self.fc = nn.Linear(hidden, n_state)

        self.noise_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x0, T_seq):
        batch, N, _ = T_seq.shape
        x0_exp = x0.unsqueeze(1).expand(-1, N, -1)
        lstm_in = torch.cat([x0_exp, T_seq], dim=-1)
        out, _ = self.lstm(lstm_in)

        out_reshaped = out.reshape(-1, out.shape[-1])
        out_norm = self.batch_norm(out_reshaped)
        out_norm = out_norm.reshape(out.shape)
        out_norm = self.dropout_layer(out_norm)

        x_hat = self.fc(out)
        return torch.nn.functional.softplus(x_hat)
