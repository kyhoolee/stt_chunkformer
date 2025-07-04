"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn


class ChunkFormerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        aggregate: int = 1
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.aggregate = aggregate
        if self.aggregate < 1:
            self.project_linear = nn.Linear(size * aggregate, size)
        else:
            self.project_linear = None
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size,
                                          eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

        
    def forward_parallel_chunk(
        self,
        x: torch.Tensor,                  # 🔹 Input tensor: (batch, time, feature_dim)
        mask: torch.Tensor,               # 🔹 Attention mask for self-attention (batch, 1, time)
        pos_emb: torch.Tensor,            # 🔹 Positional encoding (batch, time, feature_dim)
        mask_pad: torch.Tensor,           # 🔹 Padding mask for convolution (batch, 1, time)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0)),  # 🔹 Cached attention key/value (for streaming)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0)),  # 🔹 Cached CNN states (for convolution streaming)
        right_context_size: int = 0,      # 🔹 How many frames ahead this chunk can see (attention)
        left_context_size: int = 0,       # 🔹 How many frames before this chunk can see (attention)
        truncated_context_size: int = 0   # 🔹 Limit context in training (simulate streaming with truncation)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one ChunkFormer encoder layer over a streaming chunk.

        Returns:
            x: Output features (batch, time, dim)
            mask: Updated attention mask
            new_att_cache: Updated cache for attention (for next chunk)
            new_cnn_cache: Updated cache for convolution (for next chunk)
        """
        print("\n======= 🧩 [EncoderLayer.forward_parallel_chunk] START =======")
        print(f"📥 Input shape: x = {x.shape}, mask = {mask.shape}, pos_emb = {pos_emb.shape}")
        print(f"📥 Cache shapes: att_cache = {att_cache.shape}, cnn_cache = {cnn_cache.shape}")
        print(f"⚙️ Contexts: left = {left_context_size}, right = {right_context_size}, trunc = {truncated_context_size}")

        # ----------------------------------------------------------------------------------
        # 1️⃣ Macaron Feed-Forward Network (optional, giống vị trí FFN đầu trong Transformer XL)
        # ----------------------------------------------------------------------------------
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x_ff_mac = self.feed_forward_macaron(x)  # → (batch, time, dim)
            x = residual + self.ff_scale * self.dropout(x_ff_mac)
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
            print(f"🔹 After macaron FFN: x = {x.shape}")

        # ----------------------------------------------------------------------------------
        # 2️⃣ Self-Attention (streaming-aware, use cache + relative position)
        # ----------------------------------------------------------------------------------
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        # Gọi attention module theo dạng streaming chunk, dùng cache (KV trước đó)
        x_att, new_att_cache = self.self_attn.forward_parallel_chunk(
            x, x, x, mask, pos_emb, att_cache,
            right_context_size=right_context_size,
            left_context_size=left_context_size,
            truncated_context_size=truncated_context_size
        )

        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)
        print(f"🧠 After MultiHeadAttention: x = {x.shape}, new_att_cache = {new_att_cache.shape}")

        # ----------------------------------------------------------------------------------
        # 3️⃣ Convolution Module (lấy ngữ cảnh cục bộ gần – giống CNN trong CNN-Transformer)
        # ----------------------------------------------------------------------------------
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)

            x, new_cnn_cache = self.conv_module.forward_parallel_chunk(
                x, mask_pad, cnn_cache, truncated_context_size=truncated_context_size
            )

            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)
            print(f"🌊 After Convolution Module: x = {x.shape}, new_cnn_cache = {new_cnn_cache.shape}")

        # ----------------------------------------------------------------------------------
        # 4️⃣ Feed-Forward Network (cuối lớp, như chuẩn transformer)
        # ----------------------------------------------------------------------------------
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x_ff = self.feed_forward(x)  # → (batch, time, dim)
        x = residual + self.ff_scale * self.dropout(x_ff)
        if not self.normalize_before:
            x = self.norm_ff(x)
        print(f"🔸 After final FFN: x = {x.shape}")

        # ----------------------------------------------------------------------------------
        # 5️⃣ Normalize cuối nếu có conv (đảm bảo ổn định chuỗi tầng conv → FFN)
        # ----------------------------------------------------------------------------------
        if self.conv_module is not None:
            x = self.norm_final(x)
            print(f"📏 After norm_final (due to conv_module): x = {x.shape}")

        print("✅ [EncoderLayer.forward_parallel_chunk] DONE")
        return x, mask, new_att_cache, new_cnn_cache
