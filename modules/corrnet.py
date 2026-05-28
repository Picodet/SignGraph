import torch
import torch.nn as nn


class Get_Correlation(nn.Module):
    """CorrNet correlation module for 3D ResNet features.

    Args:
        channels: Number of input/output feature channels.

    Input shape:
        B, C, T, H, W

    Output shape:
        B, C, T, H, W
    """

    def __init__(self, channels):
        super().__init__()
        reduction_channel = max(channels // 16, 1)

        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)
        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)

        self.spatial_aggregation1 = nn.Conv3d(
            reduction_channel,
            reduction_channel,
            kernel_size=(9, 3, 3),
            padding=(4, 1, 1),
            groups=reduction_channel,
        )
        self.spatial_aggregation2 = nn.Conv3d(
            reduction_channel,
            reduction_channel,
            kernel_size=(9, 3, 3),
            padding=(4, 2, 2),
            dilation=(1, 2, 2),
            groups=reduction_channel,
        )
        self.spatial_aggregation3 = nn.Conv3d(
            reduction_channel,
            reduction_channel,
            kernel_size=(9, 3, 3),
            padding=(4, 3, 3),
            dilation=(1, 3, 3),
            groups=reduction_channel,
        )

        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):
        # x: B, C, T, H, W
        x2 = self.down_conv2(x)

        x_next = torch.cat([x2[:, :, 1:], x2[:, :, -1:]], dim=2)
        x_prev = torch.cat([x2[:, :, :1], x2[:, :, :-1]], dim=2)

        affinities_next = torch.einsum("bcthw,bctsd->bthwsd", x, x_next)
        affinities_prev = torch.einsum("bcthw,bctsd->bthwsd", x, x_prev)

        features_next = torch.einsum(
            "bctsd,bthwsd->bcthw",
            x_next,
            torch.sigmoid(affinities_next) - 0.5,
        )
        features_prev = torch.einsum(
            "bctsd,bthwsd->bcthw",
            x_prev,
            torch.sigmoid(affinities_prev) - 0.5,
        )
        features = features_next * self.weights2[0] + features_prev * self.weights2[1]

        x_reduced = self.down_conv(x)
        aggregated_x = (
            self.spatial_aggregation1(x_reduced) * self.weights[0]
            + self.spatial_aggregation2(x_reduced) * self.weights[1]
            + self.spatial_aggregation3(x_reduced) * self.weights[2]
        )
        aggregated_x = self.conv_back(aggregated_x)

        return features * (torch.sigmoid(aggregated_x) - 0.5)
