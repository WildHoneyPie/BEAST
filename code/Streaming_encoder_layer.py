# The code is borrowed from espnet (https://github.com/espnet/espnet)
"""Encoder definition."""
from audioop import add
import math
from typing import Optional, Tuple
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F 

from typeguard import check_argument_types


from utils import make_pad_mask
from trasformer_utils import AbsEncoder, LayerNorm, PositionwiseFeedForward, repeat
from attention import  RelPositionalEncoding, RelPositionMultiHeadedAttention




class ContextualBlockEncoderLayer(nn.Module):
    """Contexutal Block Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        total_layer_num (int): Total number of layers
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        dropout_rate,
        total_layer_num,
        pos_enc = RelPositionalEncoding
    ):
        """Construct an EncoderLayer object."""
        super(ContextualBlockEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.total_layer_num = total_layer_num
        self.pos_enc = pos_enc(size, dropout_rate)


    def forward(
        self, x, mask, x_emb=None, past_ctx=None, next_ctx=None, layer_idx=0, tempo = None, cache=None
    ):
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, block_num, hop+2, dmodel).
            x_emb (torch.Tensor):
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            past_ctx (torch.Tensor): Previous contexutal vector
            next_ctx (torch.Tensor): Next contexutal vector
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, 1, time).
            cur_ctx (torch.Tensor): Current contexutal vector
            next_ctx (torch.Tensor): Next contexutal vector
            layer_idx (int): layer index number

        """
        nbatch = x.size(0)
        nblock = x.size(1)
        if past_ctx is not None:
            if next_ctx is None:
                # store all context vectors in one tensor
                next_ctx = past_ctx.new_zeros(
                    nbatch, nblock, self.total_layer_num, x.size(-1)
                )
            else:
                x[:, :, 0] = past_ctx[:, :, layer_idx]

        # reshape ( nbatch, nblock, block_size + 2, dim )
        #     -> ( nbatch * nblock, block_size + 2, dim )
        x = x.view(-1, x.size(-2), x.size(-1))

        if x_emb is None:
            tempo = []
            x, x_emb = self.pos_enc(x)

        if mask is not None:
            mask = mask.view(-1, mask.size(-2), mask.size(-1))

        residual = x
        x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        x_ = self.dropout(self.self_attn(x_q, x, x, x_emb, mask))

        skip = x_.reshape(nbatch, -1, x_.size(-2), x_.size(-1)).squeeze(1)

        tempo.append(skip)
        x = residual + x_


        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))


        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        layer_idx += 1
        # reshape ( nbatch * nblock, block_size + 2, dim )
        #       -> ( nbatch, nblock, block_size + 2, dim )
        x = x.view(nbatch, -1, x.size(-2), x.size(-1)).squeeze(1)
        if mask is not None:
            mask = mask.view(nbatch, -1, mask.size(-2), mask.size(-1)).squeeze(1)

        if next_ctx is not None and layer_idx < self.total_layer_num:
            next_ctx[:, 0, layer_idx, :] = x[:, 0, -1, :]
            next_ctx[:, 1:, layer_idx, :] = x[:, 0:-1, -1, :]

        return x, mask, x_emb, next_ctx, next_ctx, layer_idx, tempo
        # return x, mask




class ContextualBlockTransformerEncoder(AbsEncoder):
    """Contextual Block Transformer encoder module.

    Details in Tsunoo et al. "Transformer ASR with contextual block processing"
    (https://arxiv.org/abs/1910.07204)

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of encoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        normalize_before: whether to use layer_norm before the first block
        block_size: block size for contextual block processing
        hop_Size: hop size for block processing
        look_ahead: look-ahead size for block_processing
        init_average: whether to use average as initial context (otherwise max values)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        normalize_before: bool = True,
        left_size: int = 40,
        center_size: int = 16,
        right_size: int = 16,
        init_average: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        self.rel_pos_enc = RelPositionalEncoding(d_model=output_size, dropout_rate=positional_dropout_rate)

        self.normalize_before = normalize_before
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            torch.nn.GELU()
        )
        self.encoders = repeat(
            num_blocks,
            lambda lnum: ContextualBlockEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)
            self.after_norm_t = LayerNorm(output_size)

        # for block processing
        self.left = left_size
        self.center = center_size
        self.look_ahead = right_size
        self.block_size = (left_size + center_size + right_size)
        self.init_average = init_average

    def output_size(self) -> int:
        return self._output_size


    def forward(
        self,
        xs_pad: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        B, L, D =xs_pad.shape 
        ilens = torch.ones(B)*L
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)


        # create empty output container
        total_frame_num = xs_pad.size(1)
        ys_pad = xs_pad.new_zeros(xs_pad.size())
        tempo_pad = xs_pad.new_zeros(xs_pad.size())

        past_size = self.left
        # block_size could be 0 meaning infinite
        # apply usual encoder for short sequence
        if self.block_size == 0 or total_frame_num <= self.block_size:
            xs_pad, masks, _, _, _, _, _ = self.encoders(
                xs_pad, masks, False, None, None
            )
            if self.normalize_before:
                xs_pad = self.after_norm(xs_pad)

            olens = masks.squeeze(1).sum(1)
            return xs_pad, olens, None

        # start block processing
        cur_hop = 0
        block_num = math.ceil(
            float(total_frame_num - past_size - self.look_ahead) / float(self.center)
        )

        bsize = xs_pad.size(0)  #batch_size
        addin = xs_pad.new_zeros(
            bsize, block_num, xs_pad.size(-1)
        )  # additional context embedding vecctors  (batch, block_num, d_model)
        

        # first step
        if self.init_average:  # initialize with average value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).mean(1)  #xs_pad(batch,time,d_model)
        else:  # initialize with max value
            addin[:, 0, :] = xs_pad.narrow(1, cur_hop, self.block_size).max(1)
        cur_hop += self.center
        # following steps
        while cur_hop + self.block_size < total_frame_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.center, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.center, :] = xs_pad.narrow(
                    1, cur_hop, self.block_size
                ).max(1)
            cur_hop += self.center

        # last step
        if cur_hop < total_frame_num and cur_hop // self.center < block_num:
            if self.init_average:  # initialize with average value
                addin[:, cur_hop // self.center, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).mean(1)
            else:  # initialize with max value
                addin[:, cur_hop // self.center, :] = xs_pad.narrow(
                    1, cur_hop, total_frame_num - cur_hop
                ).max(1)

        ## addin = (batch,num_block,d_model)

        # set up masks
        mask_online = xs_pad.new_zeros(
            xs_pad.size(0), block_num, self.block_size + 2, self.block_size + 2
        )
        #(batch,block_num,block+2,block+2)

        mask_online.narrow(2, 1, self.block_size + 1).narrow(
            3, 0, self.block_size + 1
        ).fill_(1)
        # mask_online = (batch,block_num, block+2, block+2)

        xs_chunk = xs_pad.new_zeros(
            bsize, block_num, self.block_size + 2, xs_pad.size(-1)
        ) #(batch, block_num, blcok+2, d_model)

        # fill the input
        # first step
        left_idx = 0
        block_idx = 0
        xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
            -2, left_idx, self.block_size
        )
        left_idx += self.center
        block_idx += 1
        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            xs_chunk[:, block_idx, 1 : self.block_size + 1] = xs_pad.narrow(
                -2, left_idx, self.block_size
            )
            left_idx += self.center
            block_idx += 1
        # last steps
        last_size = total_frame_num - left_idx
        xs_chunk[:, block_idx, 1 : last_size + 1] = xs_pad.narrow(
            -2, left_idx, last_size
        )

        # fill the initial context vector
        xs_chunk[:, 0, 0] = addin[:, 0]
        xs_chunk[:, 1:, 0] = addin[:, 0 : block_num - 1]
        xs_chunk[:, :, self.block_size + 1] = addin


        # forward
        ys_chunk, mask_online, _,_,_,_,tempo = self.encoders(
            xs_chunk, mask_online, None, xs_chunk
        )

        tempo = torch.stack(tempo, axis=-1).sum(dim=-1)

        # copy output
        # first step
        offset = self.block_size - self.look_ahead - self.center + 1
        left_idx = 0
        block_idx = 0
        cur_hop = self.block_size - self.look_ahead
        ys_pad[:, left_idx:cur_hop] = ys_chunk[:, block_idx, 1 : cur_hop + 1]
        tempo_pad[:, left_idx:cur_hop] = tempo[:, block_idx, 1 : cur_hop + 1]
        left_idx += self.center
        block_idx += 1

        # following steps
        while left_idx + self.block_size < total_frame_num and block_idx < block_num:
            ys_pad[:, cur_hop : cur_hop + self.center] = ys_chunk[
                :, block_idx, offset : offset + self.center
            ]
            tempo_pad[:, cur_hop : cur_hop + self.center] = tempo[
                :, block_idx, offset : offset + self.center
            ]
            cur_hop += self.center
            left_idx += self.center
            block_idx += 1

        ys_pad[:, cur_hop:total_frame_num] = ys_chunk[
            :, block_idx, offset : last_size + 1, :
        ]
        tempo_pad[:, cur_hop:total_frame_num] = tempo[
            :, block_idx, offset : last_size + 1, :
        ]

        if self.normalize_before:
            ys_pad = self.after_norm(ys_pad)
            tempo_pad = self.after_norm_t(tempo_pad)

        olens = masks.squeeze(1).sum(1)
        return ys_pad, olens, tempo_pad

