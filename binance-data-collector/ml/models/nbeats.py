"""
N-BEATS Model with TDA Feature Integration
Based on: "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"
https://arxiv.org/abs/1905.10437

Enhanced with TDA features per:
"Enhancing financial time series forecasting through topological data analysis"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class NBeatsBlock(nn.Module):
    """
    Basic N-BEATS Block
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon: int,
        n_layers: int = 4,
        hidden_size: int = 256,
        basis_function: str = 'generic'
    ):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.basis_function = basis_function

        # Fully connected stack
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        self.fc = nn.Sequential(*layers)

        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

        # Basis expansion
        if basis_function == 'generic':
            self.backcast_basis = nn.Linear(theta_size, input_size)
            self.forecast_basis = nn.Linear(theta_size, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, input_size)

        Returns:
            backcast: Reconstruction of input
            forecast: Prediction
        """
        # FC stack
        h = self.fc(x)

        # Theta parameters
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)

        # Basis expansion
        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)

        return backcast, forecast


class NBeatsStack(nn.Module):
    """
    Stack of N-BEATS Blocks
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_blocks: int = 3,
        n_layers: int = 4,
        hidden_size: int = 256,
        theta_size: int = 32,
        basis_function: str = 'generic'
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                theta_size=theta_size,
                horizon=horizon,
                n_layers=n_layers,
                hidden_size=hidden_size,
                basis_function=basis_function
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through stack

        Args:
            x: Input tensor

        Returns:
            backcast: Residual after all blocks
            forecast: Sum of all block forecasts
        """
        forecast = torch.zeros(x.size(0), self.blocks[0].horizon, device=x.device)
        backcast = x

        for block in self.blocks:
            b, f = block(backcast)
            backcast = backcast - b  # Residual connection
            forecast = forecast + f  # Additive forecast

        return backcast, forecast


class NBeatsWithTDA(nn.Module):
    """
    N-BEATS Model with TDA Feature Integration

    Integrates topological features (entropy, amplitude, num_points)
    into the forecasting model.
    """

    def __init__(
        self,
        lookback: int = 96,          # 8 hours of 5-min candles
        horizon: int = 12,            # 1 hour ahead
        n_stacks: int = 2,
        n_blocks: int = 3,
        n_layers: int = 4,
        hidden_size: int = 256,
        theta_size: int = 32,
        tda_features: int = 3,        # entropy, amplitude, num_points
        use_tda: bool = True
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon
        self.use_tda = use_tda
        self.tda_features = tda_features

        # TDA feature processing
        if use_tda:
            self.tda_encoder = nn.Sequential(
                nn.Linear(tda_features, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, hidden_size // 4),
                nn.ReLU()
            )
            input_size = lookback + hidden_size // 4
        else:
            input_size = lookback

        # N-BEATS Stacks
        self.stacks = nn.ModuleList([
            NBeatsStack(
                input_size=input_size,
                horizon=horizon,
                n_blocks=n_blocks,
                n_layers=n_layers,
                hidden_size=hidden_size,
                theta_size=theta_size
            )
            for _ in range(n_stacks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        tda: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Price series of shape (batch, lookback)
            tda: TDA features of shape (batch, 3)

        Returns:
            forecast: Predictions of shape (batch, horizon)
        """
        # Encode TDA features
        if self.use_tda and tda is not None:
            tda_encoded = self.tda_encoder(tda)
            x = torch.cat([x, tda_encoded], dim=1)

        # Process through stacks
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
        residual = x

        for stack in self.stacks:
            backcast, f = stack(residual)
            residual = backcast
            forecast = forecast + f

        return forecast


class NBeatsWithExogenous(nn.Module):
    """
    N-BEATS with multiple exogenous features

    Supports: TDA features, CVD, Volume, OI
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 12,
        n_stacks: int = 2,
        n_blocks: int = 3,
        n_layers: int = 4,
        hidden_size: int = 256,
        theta_size: int = 32,
        n_exog_features: int = 7,  # TDA(3) + CVD + Volume + Buy/Sell Vol
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon

        # Exogenous feature encoder
        self.exog_encoder = nn.Sequential(
            nn.Linear(n_exog_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU()
        )

        # Combined input size
        input_size = lookback + hidden_size // 4

        # N-BEATS Stacks
        self.stacks = nn.ModuleList([
            NBeatsStack(
                input_size=input_size,
                horizon=horizon,
                n_blocks=n_blocks,
                n_layers=n_layers,
                hidden_size=hidden_size,
                theta_size=theta_size
            )
            for _ in range(n_stacks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        exog: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Price series (batch, lookback)
            exog: Exogenous features (batch, n_exog_features)
        """
        exog_encoded = self.exog_encoder(exog)
        x = torch.cat([x, exog_encoded], dim=1)

        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)
        residual = x

        for stack in self.stacks:
            backcast, f = stack(residual)
            residual = backcast
            forecast = forecast + f

        return forecast


if __name__ == "__main__":
    # Test model
    batch_size = 32
    lookback = 96
    horizon = 12

    model = NBeatsWithTDA(
        lookback=lookback,
        horizon=horizon,
        use_tda=True
    )

    x = torch.randn(batch_size, lookback)
    tda = torch.randn(batch_size, 3)

    output = model(x, tda)
    print(f"Input shape: {x.shape}")
    print(f"TDA shape: {tda.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
