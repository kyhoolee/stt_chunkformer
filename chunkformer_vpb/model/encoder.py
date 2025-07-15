# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Encoder definition."""
import random
from typing import Tuple, Optional

import torch
import math


from .attention import MultiHeadedAttention
from .attention import StreamingRelPositionMultiHeadedAttention
from .convolution import ConvolutionModule
from .embedding import StreamingRelPositionalEncoding
from .encoder_layer import ChunkFormerEncoderLayer
from .positionwise_feed_forward import PositionwiseFeedForward
from .subsampling import DepthwiseConvSubsampling
from .utils.common import get_activation
from .utils.mask import make_pad_mask

class BaseEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        use_limited_chunk: bool = False,
        limited_decoding_chunk_sizes: list = [],
        limited_left_chunk_sizes: list = [],
        use_context_hint_chunk: bool = False,
        right_context_sizes: list = [],
        right_context_probs: list = [],
        freeze_subsampling_layer: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
        """
        super().__init__()
        self._output_size = output_size
        self.pos_enc_layer_type = pos_enc_layer_type
        self.attention_heads = attention_heads
        self.input_layer = input_layer


        pos_enc_class = StreamingRelPositionalEncoding
        subsampling_class = DepthwiseConvSubsampling

        self.global_cmvn = global_cmvn
        if subsampling_class == DepthwiseConvSubsampling:
            self.embed = subsampling_class(
                subsampling="dw_striding",
                subsampling_factor=8,
                feat_in=input_size,
                feat_out=output_size,
                conv_channels=output_size,
                pos_enc_class=pos_enc_class(output_size, positional_dropout_rate),
                subsampling_conv_chunking_factor=1,
                activation=torch.nn.ReLU(),
                is_causal=False,
            )
        else:
            self.embed = subsampling_class(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size * 1, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.use_limited_chunk = use_limited_chunk
        self.limited_decoding_chunk_sizes = torch.IntTensor(limited_decoding_chunk_sizes)
        self.limited_left_chunk_sizes = torch.IntTensor(limited_left_chunk_sizes)
        self.use_context_hint_chunk = use_context_hint_chunk
        self.right_context_sizes = torch.IntTensor(right_context_sizes)
        self.right_context_probs = torch.FloatTensor(right_context_probs)

        if freeze_subsampling_layer:
            self.freeze_subsampling_layer()


    def output_size(self) -> int:
        return self._output_size

    def freeze_subsampling_layer(self):
        for param in self.embed.parameters():
            param.requires_grad = False

    # def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Full-sequence (offline) forward:
    #       - xs: [B, T_in, D_in], xs_lens: [B]
    #     Trả về:
    #       - encoder_out: [B, T_out, D]
    #       - encoder_mask: [B, 1, T_out]
    #     """
    #     # 1) subsample + pos-embed
    #     xs, pos_emb, out_lens = self.embed(xs, xs_lens)
    #     # 2) tạo mask pad
    #     mask = ~make_pad_mask(out_lens, xs.size(1)).unsqueeze(1)  # [B,1,T_out]
    #     # 3) forward qua tất cả các layer
    #     for layer in self.encoders:
    #         xs, _ = layer(xs, mask, pos_emb, None, None)
    #     # 4) layer-norm sau cùng (nếu có)
    #     if self.normalize_before:
    #         xs = self.after_norm(xs)
    #     return xs, mask
    
    def forward_parallel_chunk(
        self,
        xs,
        xs_origin_lens,
        chunk_size: int = -1,
        left_context_size: int = -1,
        right_context_size: int = -1,
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        truncated_context_size: int = 0,
        offset: torch.Tensor = torch.zeros(0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        print("\n================= 🧩 [Encoder.forward_parallel_chunk] START =================")
        print(f"📥 Input shape: {xs.shape}, xs_origin_lens: {xs_origin_lens.tolist()}")
        print(f"⚙️ chunk_size={chunk_size}, left_context={left_context_size}, right_context={right_context_size}, truncated_context_size={truncated_context_size}")

        assert offset.shape[0] == len(xs), f"{offset.shape[0]} != {len(xs)}"

        # --------- Calculate chunk window size ---------
        subsampling = self.embed.subsampling_factor  # e.g., 8
        context = self.embed.right_context + 1       # current frame + right context
        size = (chunk_size - 1) * subsampling + context
        step = subsampling * chunk_size
        device = xs_origin_lens.device
        conv_lorder = self.cnn_module_kernel // 2

        print(f"📏 Subsampling: {subsampling}, Chunk frame size: {size}, Step: {step}, Conv lorder: {conv_lorder}")

        upper_bounds, lower_bounds = [], []
        upper_bounds_conv, lower_bounds_conv = [], []
        x_pad, xs_lens, n_chunks = [], [], []

        # --------- Process each sample in batch ---------
        for i, (x_len, x, offs) in enumerate(zip(xs_origin_lens, xs, offset)):
            x = x.to(device)
            original_len = x.size(0)

            # Add padding if input too short
            if x.size(0) >= size:
                n_frames_pad = (step - ((x.size(0) - size) % step)) % step
            else:
                n_frames_pad = size - x.size(0)

            x = torch.nn.functional.pad(x, (0, 0, 0, n_frames_pad))
            n_chunk = ((x.size(0) - size) // step) + 1

            # print(f"🔹 Sample {i}: original_len={original_len}, padded_len={x.size(0)}, pad_frames={n_frames_pad}, n_chunks={n_chunk}, offset={offs.item()}")

            # Unfold to overlapping windows (T, D) -> (n_chunk, D, size) -> transpose for Conv2D
            x = x.unfold(0, size=size, step=step).transpose(2, 1)  # [n_chunk, size, D] → [n_chunk, D, size]

            # -------- Compute bounds for attention & conv mask --------
            max_len = 1 + (x_len - context) // subsampling
            upper = chunk_size + right_context_size + torch.arange(0, n_chunk, device=device) * (size - context) // subsampling
            lower = upper - max_len
            upper += offs

            upper_conv = chunk_size + conv_lorder + torch.arange(0, n_chunk, device=device) * (size - context) // subsampling
            lower_conv = torch.maximum(upper_conv - max_len, torch.full_like(upper_conv, conv_lorder - right_context_size))
            upper_conv += offs

            # Save for batching
            xs_lens += [size] * (n_chunk - 1) + [size - n_frames_pad]
            upper_bounds.append(upper.unsqueeze(1))
            lower_bounds.append(lower.unsqueeze(1))
            upper_bounds_conv.append(upper_conv.unsqueeze(1))
            lower_bounds_conv.append(lower_conv.unsqueeze(1))
            x_pad.append(x)
            n_chunks.append(n_chunk)

        # --------- Stack all chunks ---------
        xs = torch.cat(x_pad, dim=0).to(device)
        xs_lens = torch.tensor(xs_lens, device=device)
        upper_bounds = torch.cat(upper_bounds).to(device)
        lower_bounds = torch.cat(lower_bounds).to(device)
        upper_bounds_conv = torch.cat(upper_bounds_conv).to(device)
        lower_bounds_conv = torch.cat(lower_bounds_conv).to(device)

        print(f"\n🧱 Total chunked xs shape: {xs.shape}")
        print(f"📐 xs_lens (post chunk): {xs_lens.shape}, total_chunks: {xs.shape[0]}")

        # --------- CMVN Normalization + Embedding ---------
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
            # print("✅ Applied Global CMVN")

        xs, pos_emb, xs_lens = self.embed(xs, xs_lens, offset=left_context_size, right_context_size=right_context_size)
        print(f"🎛️ Embedded xs shape: {xs.shape}, PosEmb shape: {pos_emb.shape}")

        # --------- Create attention masks ---------
        mask_pad_idx = torch.arange(0, conv_lorder + chunk_size + conv_lorder, device=device).unsqueeze(0).repeat(xs.size(0), 1)
        mask_pad = (lower_bounds_conv <= mask_pad_idx) & (mask_pad_idx < upper_bounds_conv)
        mask_pad = mask_pad.flip(-1).unsqueeze(1)

        att_mask_idx = torch.arange(0, left_context_size + chunk_size + right_context_size, device=device).unsqueeze(0).repeat(xs.size(0), 1)
        att_mask = (lower_bounds <= att_mask_idx) & (att_mask_idx < upper_bounds)
        att_mask = att_mask.flip(-1).unsqueeze(1)

        print(f"🧮 att_mask shape: {att_mask.shape}, mask_pad shape: {mask_pad.shape}")

        # --------- Forward through all encoder layers ---------
        r_att_cache, r_cnn_cache = [], []
        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer.forward_parallel_chunk(
                xs, att_mask, pos_emb,
                mask_pad=mask_pad,
                right_context_size=right_context_size,
                left_context_size=left_context_size,
                att_cache=att_cache[i].to(device) if att_cache.size(0) > 0 else att_cache,
                cnn_cache=cnn_cache[i].to(device) if cnn_cache.size(0) > 0 else cnn_cache,
                truncated_context_size=truncated_context_size
            )
            r_att_cache.append(new_att_cache)
            r_cnn_cache.append(new_cnn_cache)
            print(f"🧩 Layer {i}: xs shape after layer = {xs.shape}")
            # print(f"\t🧩 Layer {i}\n\t\t{layer}")

        # --------- Final normalization and output ---------
        if self.normalize_before:
            xs = self.after_norm(xs)
            print("📏 Applied LayerNorm after encoder")

        xs_lens = self.embed.calc_length(xs_origin_lens)
        offset += xs_lens
        print(f"📤 Final offset: {offset.tolist()}")

        r_att_cache = torch.stack(r_att_cache, dim=0)
        r_cnn_cache = torch.stack(r_cnn_cache, dim=0)

        print(f"\n✅ [Encoder Output] xs: {xs.shape}, xs_lens: {xs_lens.tolist()}, n_chunks: {n_chunks}")
        print("====================================================================\n")
        return xs, xs_lens, n_chunks, r_att_cache, r_cnn_cache, offset

    def ctc_forward(self, xs, xs_lens=None, n_chunks=None):
        """
        Perform greedy decoding on encoder output using CTC.

        Args:
            xs: Tensor [B, T, D] 
                - Encoder output: batch of sequences
                - B = batch size
                - T = max time steps (after subsampling)
                - D = encoder feature dim (e.g., 512)
            
            xs_lens: Optional[List[int]]
                - Độ dài thực tế của mỗi chuỗi trong batch (sau subsample)
            
            n_chunks: Optional[int]
                - Nếu bạn xử lý đầu vào từ các chunk (streaming), có thể chia theo chunk tại đây

        Returns:
            framewise_token_ids: 
                - Nếu không có chunk: Tensor [B, T] — ID token top-1 mỗi timestep
                - Nếu có chunk: List[Tensor[T_i]] — mỗi phần là 1 sequence từ chunk
        """

        # === Step 1: Tính log-softmax (log xác suất) trên toàn vocab tại mỗi frame ===
        # log_probs: [B, T, vocab_size]
        log_probs = self.ctc.log_softmax(xs)

        # === Step 2: Greedy decode (top-1 theo chiều vocab tại mỗi frame) ===
        # top1_index: [B, T, 1] — chứa chỉ số nhãn có log_prob cao nhất tại mỗi thời điểm
        top1_logprob, top1_index = log_probs.topk(1, dim=2)

        # === Step 3: Bỏ chiều cuối cùng để thu được chuỗi nhãn theo frame ===
        # framewise_token_ids: [B, T] — mỗi phần tử là ID nhãn (int) tại từng frame
        framewise_token_ids = top1_index.squeeze(-1)

        # === Step 4: Nếu xử lý theo chunk, chia batch thành các đoạn tương ứng ===
        if (n_chunks is not None) and (xs_lens is not None):
            # split: chia tensor theo batch dim thành list có độ dài n_chunks
            framewise_token_ids = framewise_token_ids.split(n_chunks, dim=0)

            # Cắt từng chunk theo độ dài thực tế xs_lens (tránh phần padding)
            # Output: List[Tensor[T_i]] — mỗi phần là 1 chuỗi ID token
            framewise_token_ids = [
                token_ids.flatten()[:x_len] for token_ids, x_len in zip(framewise_token_ids, xs_lens)
            ]

        return framewise_token_ids



    def rearrange(
        self, 
        xs,
        xs_lens,
        n_chunks
    ):
        xs = xs.split(n_chunks, dim=0)   
        xs_lens = self.embed.calc_length(xs_lens)
        xs = [x.reshape(-1, self._output_size)[:x_len] for x, x_len in zip(xs, xs_lens)]



        xs = torch.nn.utils.rnn.pad_sequence(xs,
                                    batch_first=True,
                                    padding_value=0)
        masks = ~make_pad_mask(xs_lens, xs.size(1)).unsqueeze(1).to(xs.device) # (B, 1, T)
        return xs, masks

class ChunkFormerEncoder(BaseEncoder):
    """ChunkFormer encoder module."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        use_limited_chunk: bool = False,
        limited_decoding_chunk_sizes: list = [],
        limited_left_chunk_sizes: list = [],
        use_dynamic_conv: bool = False,
        use_context_hint_chunk: bool = False,
        right_context_sizes: list = [],
        right_context_probs: list = [],
        freeze_subsampling_layer: bool = False,
    ):
        """Construct ChunkFormerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk, 
                         use_limited_chunk=use_limited_chunk,
                         limited_decoding_chunk_sizes=limited_decoding_chunk_sizes,
                         limited_left_chunk_sizes=limited_left_chunk_sizes,
                         use_context_hint_chunk=use_context_hint_chunk,
                         right_context_sizes=right_context_sizes,
                         right_context_probs=right_context_probs,
                         freeze_subsampling_layer=freeze_subsampling_layer)
        self.cnn_module_kernel = cnn_module_kernel
        activation = get_activation(activation_type)
        self.num_blocks = num_blocks
        self.use_dynamic_conv = use_dynamic_conv
        self.input_size = input_size
        self.attention_heads = attention_heads

        # self-attention module definition
        if pos_enc_layer_type == "abs_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        elif pos_enc_layer_type == "rel_pos":
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        elif pos_enc_layer_type == "stream_rel_pos":
            encoder_selfattn_layer = StreamingRelPositionMultiHeadedAttention
        
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )

        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal, True, use_dynamic_conv)

        self.encoders = torch.nn.ModuleList([
            ChunkFormerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                aggregate=2 if ((i % 3 == 0) and  (i > 0)) else 1
            ) for i in range(num_blocks)
        ])

    def limited_context_selection(self):
        full_context_training = True
        if (self.dynamic_chunk_sizes is not None
            and self.dynamic_left_context_sizes is not None
                and self.dynamic_right_context_sizes is not None):
            chunk_size = random.choice(self.dynamic_chunk_sizes)
            left_context_size = random.choice(self.dynamic_left_context_sizes)
            right_context_size = random.choice(self.dynamic_right_context_sizes)
            full_context_training = not (chunk_size > 0
                                         and left_context_size > 0
                                         and right_context_size > 0)

        if full_context_training:
            chunk_size, left_context_size, right_context_size = 0, 0, 0
        return chunk_size, left_context_size, right_context_size

    def forward_encoder(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        chunk_size: int = 0,
        left_context_size: int = 0,
        right_context_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            chunk_size (int): Chunk size for limited chunk context
            left_context_size (int): Left context size for limited chunk context
            right_context_size (int): Right context size for limited chunk context
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, masks = self.embed(
            xs, masks,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size
        )
        mask_pad = masks  # (B, 1, T/subsample_rate)

        xs = self.forward_layers(
            xs, masks, pos_emb, mask_pad,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
        )
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor,
                       chunk_size: int = 0,
                       left_context_size: int = 0,
                       right_context_size: int = 0) -> torch.Tensor:
        for idx, layer in enumerate(self.encoders):
            xs, chunk_masks, _, _ = layer(
                xs, chunk_masks, pos_emb, mask_pad,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
            )
        return xs

    def forward(self,
                xs: torch.Tensor,
                xs_lens: torch.Tensor,
                decoding_chunk_size: int = 0,
                num_decoding_left_chunks: int = -1,
                **kwargs):
        """
        Main forward function that dispatches to either the standard
        forward pass or the parallel chunk version based on the
        model's training mode.
        """
        # for masked batch chunk context inference
        # should add a better flag to trigger
        if decoding_chunk_size > 0 and num_decoding_left_chunks > 0:
            # If both decoding_chunk_size and num_decoding_left_chunks
            # are set, use the parallel chunk decoding.
            return self.forward_parallel_chunk(
                xs=xs,
                xs_origin_lens=xs_lens,
                chunk_size=decoding_chunk_size,
                left_context_size=num_decoding_left_chunks,
                # we assume left and right context are the same
                right_context_size=num_decoding_left_chunks,
                **kwargs
            )
        else:
            (chunk_size,
                left_context_size,
                right_context_size) = self.limited_context_selection()
            return self.forward_encoder(
                xs=xs,
                xs_lens=xs_lens,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                **kwargs
            )
