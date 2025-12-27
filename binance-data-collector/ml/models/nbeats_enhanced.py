"""
Enhanced N-BEATS Model with Futures-Spot Correlation Features
"""
import torch
import torch.nn as nn
import numpy as np


class NBeatsBlock(nn.Module):
    """Single N-BEATS block"""

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon: int,
        n_neurons: int = 512,
        n_layers: int = 4,
        share_weights: bool = False
    ):
        super().__init__()

        self.input_size = input_size
        self.horizon = horizon
        self.theta_size = theta_size

        # Fully connected stack
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_size if i == 0 else n_neurons, n_neurons))
            layers.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*layers)

        # Theta parameters
        self.theta_b = nn.Linear(n_neurons, input_size)  # Backcast (same size as input)
        self.theta_f = nn.Linear(n_neurons, horizon)  # Forecast

    def forward(self, x):
        # x shape: (batch, input_size)
        h = self.fc_stack(x)

        backcast = self.theta_b(h)
        forecast = self.theta_f(h)

        return backcast, forecast


class FuturesSpotAttention(nn.Module):
    """
    Attention mechanism to weigh futures-spot correlation features
    """

    def __init__(self, feature_dim: int):
        super().__init__()

        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.scale = np.sqrt(feature_dim)

    def forward(self, x):
        # x shape: (batch, feature_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Self-attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended = torch.matmul(attention_weights, V)

        return attended + x  # Residual connection


class NBeatsEnhanced(nn.Module):
    """
    Enhanced N-BEATS with Futures-Spot Correlation Features

    Features:
    - Price spread (futures - spot)
    - Volume ratio
    - CVD difference
    - Return correlation
    - Spread momentum
    """

    def __init__(
        self,
        lookback: int = 96,
        horizon: int = 12,
        n_correlation_features: int = 8,  # Futures-spot correlation features
        n_exog_features: int = 0,  # TDA + other features
        n_stacks: int = 3,
        n_blocks_per_stack: int = 2,
        n_neurons: int = 512,
        n_layers: int = 4,
        use_attention: bool = True
    ):
        super().__init__()

        self.lookback = lookback
        self.horizon = horizon
        self.n_correlation_features = n_correlation_features
        self.use_attention = use_attention

        # Correlation feature processor with attention
        if n_correlation_features > 0:
            self.correlation_encoder = nn.Sequential(
                nn.Linear(n_correlation_features, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU()
            )

            if use_attention:
                self.correlation_attention = FuturesSpotAttention(32)

        # Exogenous feature processor (TDA, etc.)
        if n_exog_features > 0:
            self.exog_encoder = nn.Sequential(
                nn.Linear(n_exog_features, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU()
            )

        # Calculate total input size
        base_input_size = lookback
        if n_correlation_features > 0:
            base_input_size += 32  # Encoded correlation features
        if n_exog_features > 0:
            base_input_size += 16  # Encoded exog features

        # N-BEATS stacks
        self.stacks = nn.ModuleList()

        for _ in range(n_stacks):
            stack_blocks = nn.ModuleList()
            for _ in range(n_blocks_per_stack):
                block = NBeatsBlock(
                    input_size=base_input_size,
                    theta_size=horizon,
                    horizon=horizon,
                    n_neurons=n_neurons,
                    n_layers=n_layers
                )
                stack_blocks.append(block)
            self.stacks.append(stack_blocks)

        # Final projection
        self.forecast_head = nn.Linear(horizon * n_stacks, horizon)

    def forward(self, x, correlation_features=None, exog_features=None):
        """
        Args:
            x: Historical prices (batch, lookback)
            correlation_features: Futures-spot correlation features (batch, n_correlation_features)
            exog_features: TDA and other exogenous features (batch, n_exog_features)

        Returns:
            forecast: Predicted prices (batch, horizon)
        """
        batch_size = x.size(0)

        # Encode correlation features
        if self.n_correlation_features > 0 and correlation_features is not None:
            corr_encoded = self.correlation_encoder(correlation_features)

            if self.use_attention:
                corr_encoded = self.correlation_attention(corr_encoded)

            # Concatenate with price history
            x = torch.cat([x, corr_encoded], dim=1)

        # Encode exogenous features
        if exog_features is not None:
            exog_encoded = self.exog_encoder(exog_features)
            x = torch.cat([x, exog_encoded], dim=1)

        # Process through N-BEATS stacks
        forecast_stack = []

        for stack in self.stacks:
            for block in stack:
                backcast, forecast = block(x)
                x = x - backcast  # Residual learning
                forecast_stack.append(forecast)

        # Combine forecasts from all stacks
        combined_forecast = torch.cat(forecast_stack, dim=1)
        final_forecast = self.forecast_head(combined_forecast)

        return final_forecast


if __name__ == "__main__":
    # Test model
    print("Testing NBeatsEnhanced...")

    model = NBeatsEnhanced(
        lookback=96,
        horizon=12,
        n_correlation_features=8,
        n_exog_features=3,
        use_attention=True
    )

    # Create dummy input
    batch_size = 32
    x = torch.randn(batch_size, 96)  # Price history
    corr_features = torch.randn(batch_size, 8)  # Correlation features
    exog_features = torch.randn(batch_size, 3)  # TDA features

    # Forward pass
    output = model(x, corr_features, exog_features)

    print(f"Input shape: {x.shape}")
    print(f"Correlation features shape: {corr_features.shape}")
    print(f"Exog features shape: {exog_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✓ Model test passed!")
