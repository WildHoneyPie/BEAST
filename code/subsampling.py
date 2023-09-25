"""Subsampling layer definition."""

import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
import torch.nn as nn


class MusicSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(MusicSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 3), stride=1, padding=(2, 0)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 12), stride=1, padding=(0, 0)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=odim, kernel_size=(3, 6), stride=1, padding=(1, 0)),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )


    def forward(self, x):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, idim)
        x = self.conv(x)
        print(x.shape)
        x = x.transpose(1, 3).squeeze(1).contiguous()    #(batch, time, channel=dmodel)

        return x

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]