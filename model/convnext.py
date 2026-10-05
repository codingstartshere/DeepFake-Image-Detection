"""
ConvNeXt-Tiny model for CIFAKE image deepfake classification.

Project location:
    model/convnext.py

The model is designed for small CIFAKE images and provides two outputs:
    1. logits   -> binary Real/Fake classification
    2. features -> deep feature vector before the classification head

Example:
    from model.convnext import ConvNeXtTiny

    model = ConvNeXtTiny(num_classes=2)
    logits, features = model(images, return_features=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm applied over the channel dimension of a 4D tensor."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N, C, H, W -> N, H, W, C
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x,
            (x.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )
        # N, H, W, C -> N, C, H, W
        return x.permute(0, 3, 1, 2)


class LayerNorm(nn.Module):
    """LayerNorm over the channel dimension for channels-last tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            (x.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block.

    Structure:
        Depthwise 7x7 convolution
        -> LayerNorm
        -> Linear expansion (4x)
        -> GELU
        -> Linear projection
        -> Layer Scale
        -> Residual connection
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        self.norm = LayerNorm(dim)

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)

        # N, C, H, W -> N, H, W, C
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x

        # N, H, W, C -> N, C, H, W
        x = x.permute(0, 3, 1, 2)

        return residual + self.drop_path(x)


class DropPath(nn.Module):
    """Stochastic depth / drop path."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )

        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny adapted for CIFAKE binary classification.

    Default architecture:
        depths = [3, 3, 9, 3]
        dims   = [96, 192, 384, 768]

    The classifier outputs two logits:
        class 0 -> Real
        class 1 -> Fake

    The model can also return the 768-dimensional deep feature vector
    immediately before the classifier.

    Parameters
    ----------
    num_classes : int
        Number of output classes. CIFAKE uses 2.
    in_channels : int
        Number of input image channels. RGB = 3.
    drop_path_rate : float
        Maximum stochastic-depth probability.
    dropout : float
        Dropout probability before the classifier.
    layer_scale_init_value : float
        Initial value for ConvNeXt layer scaling.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 3,
        drop_path_rate: float = 0.1,
        dropout: float = 0.2,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feature_dim = 768

        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        # Stem.
        # CIFAKE images are small, so the initial 4x4 / stride-4
        # ConvNeXt stem is retained without resizing the images here.
        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        # Three subsequent downsampling stages.
        for i in range(3):
            downsample = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample)

        # Stochastic-depth probabilities distributed through all blocks.
        total_blocks = sum(depths)
        drop_path_rates = torch.linspace(
            0.0,
            drop_path_rate,
            total_blocks,
        ).tolist()

        self.stages = nn.ModuleList()

        block_index = 0
        for stage_index in range(4):
            blocks = []

            for _ in range(depths[stage_index]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[stage_index],
                        drop_path=drop_path_rates[block_index],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                block_index += 1

            self.stages.append(nn.Sequential(*blocks))

        self.norm = nn.LayerNorm(self.feature_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the deep feature vector before classification."""

        for downsample, stage in zip(
            self.downsample_layers,
            self.stages,
        ):
            x = downsample(x)
            x = stage(x)

        # Global average pooling.
        x = x.mean(dim=(-2, -1))

        # Final LayerNorm.
        x = self.norm(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch, 3, H, W].
        return_features : bool
            If True, returns (logits, features).
            If False, returns logits only.

        Returns
        -------
        torch.Tensor
            Classification logits.

        OR

        tuple[torch.Tensor, torch.Tensor]
            (logits, deep_feature_vector)
        """

        features = self.forward_features(x)
        logits = self.head(features)

        if return_features:
            return logits, features

        return logits


def create_model(
    num_classes: int = 2,
    in_channels: int = 3,
    **kwargs,
) -> ConvNeXtTiny:
    """Factory function for creating the CIFAKE ConvNeXt-Tiny model."""

    return ConvNeXtTiny(
        num_classes=num_classes,
        in_channels=in_channels,
        **kwargs,
    )


if __name__ == "__main__":
    # Basic architecture test.
    # CIFAKE images are 32x32 RGB images.
    model = ConvNeXtTiny(num_classes=2)

    dummy_input = torch.randn(4, 3, 32, 32)

    logits, features = model(
        dummy_input,
        return_features=True,
    )

    print("Model: ConvNeXt-Tiny")
    print(f"Input shape:    {tuple(dummy_input.shape)}")
    print(f"Feature shape:  {tuple(features.shape)}")
    print(f"Logits shape:   {tuple(logits.shape)}")
